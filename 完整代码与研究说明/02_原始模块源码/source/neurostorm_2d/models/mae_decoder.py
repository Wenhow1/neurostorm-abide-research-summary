"""二维 MAE 解码器：从 4×4 瓶颈逐级扩展并预测每个 patch 的像素。"""

from __future__ import annotations

from torch import Tensor, nn

from .patches import _pair


class PatchExpand2D(nn.Module):
    """将 BHWC 网格的每个 token 扩展为 2×2 个较细的 token。"""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        if in_dim < 1 or out_dim < 1:
            raise ValueError("in_dim and out_dim must be positive")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, 4 * out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, grid: Tensor) -> Tensor:
        """[B,H,W,C] → [B,2H,2W,C_out]，保持二维相邻关系。"""
        if grid.ndim != 4 or grid.shape[-1] != self.in_dim:
            raise ValueError(
                f"expected [B,H,W,{self.in_dim}], got {tuple(grid.shape)}"
            )
        batch, height, width, _ = grid.shape
        expanded = self.proj(grid).reshape(
            batch, height, width, 2, 2, self.out_dim
        )
        expanded = expanded.permute(0, 1, 3, 2, 4, 5).reshape(
            batch, 2 * height, 2 * width, self.out_dim
        )
        return self.norm(expanded)


class MAEDecoder2D(nn.Module):
    """4²×256 → 8²×128 → 16²×64 → 32²×32 → 1024×16。

    此处没有来自未遮挡 patch 嵌入的跳接，避免把重建目标泄漏给预测头。
    最后的 16 是单通道 4×4 patch 的像素数，不含 sigmoid 或值域截断。
    """

    def __init__(
        self,
        *,
        stage_dims: tuple[int, int, int, int] = (256, 128, 64, 32),
        input_grid: tuple[int, int] = (4, 4),
        patch_size: int | tuple[int, int] = 4,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        if len(stage_dims) != 4 or any(dim < 1 for dim in stage_dims):
            raise ValueError("stage_dims must contain four positive dimensions")
        if out_channels < 1:
            raise ValueError("out_channels must be positive")
        self.stage_dims = stage_dims
        self.input_grid = _pair(input_grid, "input_grid")
        self.patch_size = _pair(patch_size, "patch_size")
        self.out_channels = out_channels

        self.expand_4_to_8 = PatchExpand2D(stage_dims[0], stage_dims[1])
        self.expand_8_to_16 = PatchExpand2D(stage_dims[1], stage_dims[2])
        self.expand_16_to_32 = PatchExpand2D(stage_dims[2], stage_dims[3])
        patch_values = out_channels * self.patch_size[0] * self.patch_size[1]
        self.pixel_head = nn.Linear(stage_dims[3], patch_values)

    @property
    def output_grid(self) -> tuple[int, int]:
        return self.input_grid[0] * 8, self.input_grid[1] * 8

    def forward(self, bottleneck: Tensor) -> Tensor:
        """[B,4,4,256] → [B,1024,16]，默认配置对应 128×128 图。"""
        expected = (*self.input_grid, self.stage_dims[0])
        if bottleneck.ndim != 4 or tuple(bottleneck.shape[1:]) != expected:
            raise ValueError(
                f"expected bottleneck [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(bottleneck.shape)}"
            )
        grid = self.expand_4_to_8(bottleneck)
        grid = self.expand_8_to_16(grid)
        grid = self.expand_16_to_32(grid)
        batch, height, width, _ = grid.shape
        return self.pixel_head(grid).reshape(batch, height * width, -1)
