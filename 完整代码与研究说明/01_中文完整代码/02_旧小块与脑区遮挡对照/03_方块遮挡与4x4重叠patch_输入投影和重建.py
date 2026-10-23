# 共同方块mask与小块输入重建实现
# 作用：ROI模型复用本文件sample_mask；旧小块还使用几何gather和average重建。
# 输入输出：FC[B,1,116,116]；mask同shape；旧小块gather后单token16值→投影宽度；最后回到FC。
# 参数与配置：mask_unit4；paired_symmetric；先移除隐藏真值再做重叠patch；只评分隐藏非对角位置。
# 注意：此基础类默认宽度16等属于较早基类配置；E1实际配置由机制模块覆盖为36起始。不要把基类默认值当E1实跑参数。
# 来源模块：source/neurostorm_2d/models/overlap_mae.py
# 原始源码镜像：代码核验/原始源码/source/neurostorm_2d/models/overlap_mae.py
# 源文件SHA256：e900e18d1270f43ec18a027e3fbf57303cb71c7f9b075ff9fe7c1b8d6ed0fb75
# 完整性：本文件保留来源模块全部可执行语句，仅格式化并增加中文注释；Python AST逐项一致。
# 运行边界：这是保留原import名称的代码阅读版本。中文文件名不能替代原模块路径；运行需原数据、划分、权重、软件环境与原工程布局。未声称独立可运行。
# 副作用提醒：含main或顶层训练/提取语句的脚本若直接执行会读写文件或使用CUDA。本次整理未执行模型。

"""Native-size FC MAE. Mask original entries before any overlapping embedding.

This version is intentionally separate from the historical padded architecture.
Its checkpoints and loss geometry are not interchangeable with that model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .mae import sample_patch_mask
from .mamba_block import MambaFactory
from .overlap_geometry import GridGroups2D
from .overlap_backbone import OverlapSWMEncoder2D, OverlapMAEDecoder2D


class OverlapMaskedAutoencoder2D(nn.Module):
    def __init__(
        self,
        *,
        image_size: tuple[int, int] = (116, 116),
        patch_size: int = 4,
        stride: int = 2,
        embed_dim: int = 16,
        depths: Sequence[int] = (2, 2, 1, 1),
        window_sizes: Sequence[int] = (8, 8, 8, 8),
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.5,
        mask_policy: str = "paired_symmetric",
        mask_unit: int = 4,
        mamba_factory: MambaFactory | None = None,
    ) -> None:
        super().__init__()
        if image_size[0] != image_size[1] or image_size[0] % mask_unit:
            raise ValueError("symmetric image must be divisible by the fixed mask unit")
        if mask_policy != "paired_symmetric":
            raise ValueError("overlap FC MAE requires paired_symmetric masks")
        if not math.isfinite(mask_ratio) or not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio must lie strictly between zero and one")
        self.image_size = tuple(image_size)
        self.patch_size, self.stride = patch_size, stride
        self.embed_dim, self.mask_unit = embed_dim, mask_unit
        self.mask_ratio, self.mask_policy = mask_ratio, mask_policy
        self.geometry = GridGroups2D(
            self.image_size, patch_size, stride, edge_align=False
        )
        self.value_projection = nn.Linear(patch_size**2, embed_dim)
        # The mask pattern distinguishes a missing value from an observed zero.
        self.mask_projection = nn.Linear(patch_size**2, embed_dim, bias=False)
        self.encoder = OverlapSWMEncoder2D(
            embed_dim=embed_dim,
            depths=depths,
            window_sizes=window_sizes,
            grid_size=self.geometry.output_shape,
            mlp_ratio=mlp_ratio,
            mamba_factory=mamba_factory,
        )
        self.decoder = OverlapMAEDecoder2D(
            stage_shapes=self.encoder.stage_shapes,
            stage_dims=self.encoder.stage_dims,
            patch_area=patch_size**2,
        )
        diagonal = torch.eye(image_size[0], dtype=torch.bool).reshape(1, 1, *image_size)
        self.register_buffer("off_diagonal", ~diagonal, persistent=False)

    def sample_mask(
        self,
        batch: int,
        *,
        device: torch.device,
        generator: torch.Generator | None = None,
        mask_ratio: float | None = None,
    ) -> Tensor:
        side = self.image_size[0] // self.mask_unit
        valid = torch.ones(batch, side * side, dtype=torch.bool, device=device)
        blocks = sample_patch_mask(
            valid,
            self.mask_ratio if mask_ratio is None else mask_ratio,
            generator=generator,
            policy=self.mask_policy,
        )
        return (
            blocks.reshape(batch, 1, side, side)
            .repeat_interleave(self.mask_unit, 2)
            .repeat_interleave(self.mask_unit, 3)
        )

    def embedded_input(self, image: Tensor, pixel_mask: Tensor) -> Tensor:
        if image.ndim != 4 or tuple(image.shape[1:]) != (1, *self.image_size):
            raise ValueError("image must be BCHW with the configured native FC shape")
        if not torch.is_floating_point(image) or not torch.isfinite(image).all():
            raise ValueError("this version requires finite floating-point FC inputs")
        if pixel_mask.shape != image.shape or pixel_mask.dtype != torch.bool:
            raise ValueError("pixel_mask must be a boolean tensor with the input shape")
        if pixel_mask.device != image.device:
            raise ValueError("image and mask must share a device")
        if not torch.equal(pixel_mask, pixel_mask.transpose(-1, -2)):
            raise ValueError("pixel_mask must hide transpose pairs together")
        # No operation depending on hidden values is permitted before this line.
        hidden_input = image.masked_fill(pixel_mask, 0).permute(0, 2, 3, 1)
        values = self.geometry.gather(hidden_input).squeeze(-1)
        pattern = self.geometry.gather(
            pixel_mask.permute(0, 2, 3, 1).to(image.dtype)
        ).squeeze(-1)
        tokens = self.value_projection(values) + self.mask_projection(pattern)
        return tokens.reshape(
            image.shape[0], *self.geometry.output_shape, self.embed_dim
        )

    def encode_full(self, image: Tensor, *, use_prompts: bool = False) -> Tensor:
        """Downstream input remains full FC; this path never accepts diagnosis labels."""
        tokens = self.embedded_input(image, torch.zeros_like(image, dtype=torch.bool))
        return self.encoder(tokens, use_prompts=use_prompts)

    def forward(
        self,
        image: Tensor,
        *,
        pixel_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        if pixel_mask is None:
            pixel_mask = self.sample_mask(
                image.shape[0], device=image.device, generator=generator
            )
        tokens = self.embedded_input(image, pixel_mask)
        patches = self.decoder(self.encoder(tokens))
        prediction = self.geometry.average(patches.unsqueeze(-1)).permute(0, 3, 1, 2)
        score_mask = pixel_mask & self.off_diagonal
        if not torch.all(score_mask.flatten(1).any(1)):
            raise ValueError(
                "each subject needs at least one hidden off-diagonal target"
            )
        return {
            "prediction": prediction,
            "target": image.detach(),
            "pixel_mask": pixel_mask,
            "score_mask": score_mask,
            "pred_patches": patches,
        }


def overlap_masked_mse(output: dict[str, Tensor]) -> Tensor:
    """Every hidden original off-diagonal entry contributes exactly once."""
    prediction, target, selected = (
        output["prediction"],
        output["target"],
        output["score_mask"],
    )
    if prediction.shape != target.shape or selected.shape != target.shape:
        raise ValueError("prediction, target and score_mask must have the same shape")
    if selected.dtype != torch.bool or not selected.any():
        raise ValueError("score_mask must select some targets")
    # Select before arithmetic: unscored entries must contribute neither loss
    # nor a NaN derivative through a nominally zero upstream gradient.
    return (
        (prediction.masked_select(selected) - target.masked_select(selected))
        .square()
        .mean()
    )
