"""二维矩阵的可逆分块和线性 patch 嵌入。"""

from __future__ import annotations

from torch import Tensor, nn


def _pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        result = (value, value)
    elif isinstance(value, tuple) and len(value) == 2 and all(
        isinstance(item, int) for item in value
    ):
        result = value
    else:
        raise ValueError(f"{name} must be a positive int or a pair of positive ints")
    if result[0] < 1 or result[1] < 1:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def patchify(image: Tensor, patch_size: int | tuple[int, int] = 4) -> Tensor:
    """[B,C,H,W] → [B,(H/ph)(W/pw),C*ph*pw]，按行优先排列块。

    块内顺序为通道、行、列；输入高宽必须整除块大小，不做隐式补边。
    对 128×128 单通道输入及 p=4，输出 [B,1024,16]。
    """
    ph, pw = _pair(patch_size, "patch_size")
    if image.ndim != 4:
        raise ValueError(f"image must be [B,C,H,W], got {tuple(image.shape)}")
    batch, channels, height, width = image.shape
    if min(batch, channels, height, width) < 1:
        raise ValueError("image dimensions must be positive")
    if height % ph or width % pw:
        raise ValueError(
            f"image {(height, width)} must be divisible by patch_size {(ph, pw)}"
        )
    grid_h, grid_w = height // ph, width // pw
    return (
        image.reshape(batch, channels, grid_h, ph, grid_w, pw)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, grid_h * grid_w, channels * ph * pw)
    )


def unpatchify(
    patches: Tensor,
    *,
    image_size: tuple[int, int],
    patch_size: int | tuple[int, int] = 4,
    channels: int = 1,
) -> Tensor:
    """[B,N,C*ph*pw] → [B,C,H,W]，精确逆转 patchify 的排列。"""
    ph, pw = _pair(patch_size, "patch_size")
    height, width = _pair(image_size, "image_size")
    if patches.ndim != 3:
        raise ValueError(f"patches must be [B,N,V], got {tuple(patches.shape)}")
    if not isinstance(channels, int) or channels < 1:
        raise ValueError("channels must be a positive int")
    if height % ph or width % pw:
        raise ValueError(
            f"image_size {(height, width)} must be divisible by patch_size {(ph, pw)}"
        )
    batch, count, values = patches.shape
    grid_h, grid_w = height // ph, width // pw
    if batch < 1 or count != grid_h * grid_w or values != channels * ph * pw:
        raise ValueError(
            f"expected patches [B,{grid_h * grid_w},{channels * ph * pw}], "
            f"got {tuple(patches.shape)}"
        )
    return (
        patches.reshape(batch, grid_h, grid_w, channels, ph, pw)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, channels, height, width)
    )


class PatchEmbed2D(nn.Module):
    """把每个非重叠像素块线性映射为一个 token，输出 BHWC 网格。"""

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 32,
        patch_size: int | tuple[int, int] = 4,
    ) -> None:
        super().__init__()
        if not isinstance(in_channels, int) or in_channels < 1:
            raise ValueError("in_channels must be a positive int")
        if not isinstance(embed_dim, int) or embed_dim < 1:
            raise ValueError("embed_dim must be a positive int")
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = _pair(patch_size, "patch_size")
        self.proj = nn.Linear(in_channels * self.patch_size[0] * self.patch_size[1], embed_dim)

    def forward(self, image: Tensor) -> Tensor:
        """[B,C,H,W] → [B,H/ph,W/pw,D]；128×128/p4/D32 → [B,32,32,32]。"""
        if image.ndim != 4:
            raise ValueError(f"image must be [B,C,H,W], got {tuple(image.shape)}")
        if image.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {image.shape[1]}"
            )
        ph, pw = self.patch_size
        patches = patchify(image, self.patch_size)
        return self.proj(patches).reshape(
            image.shape[0], image.shape[2] // ph, image.shape[3] // pw, self.embed_dim
        )
