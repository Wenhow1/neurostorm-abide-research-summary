"""将 patch 嵌入、有效区域遮挡、二维 SWM 编码器和重建解码器组合。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from neurostorm_2d.preprocessing.padding import pad_to_square

from .mae_decoder import MAEDecoder2D
from .patches import PatchEmbed2D, _pair, patchify


def sample_patch_mask(
    valid_patch_mask: Tensor,
    mask_ratio: float,
    *,
    generator: torch.Generator | None = None,
    policy: str = "independent",
) -> Tensor:
    """只从有效 patch 采样；每个样本至少留一个可见、遮一个有效块。

    输入/输出均为 [B,N] bool。标准 116→128、p=4 时，有效块数为
    841；mask_ratio=0.5 取 floor(841×0.5)=420 块，实际比率 420/841。
    固定 generator 或直接传入固定 patch_mask 可复现验证集遮挡。
    paired_symmetric 将 (i,j)/(j,i) 两块同时遮挡。对角线块是单独的
    遮挡单位；当有效区为 29×29、比例为 0.5 时，遮挡 203 对非对角块
    和 14 个对角块，恰好 420 块。此策略只适用于对称的方形有效区。
    """
    if valid_patch_mask.ndim != 2 or valid_patch_mask.dtype != torch.bool:
        raise ValueError("valid_patch_mask must be bool [B,N]")
    if not math.isfinite(mask_ratio) or not 0 < mask_ratio < 1:
        raise ValueError("mask_ratio must be finite and strictly between 0 and 1")
    batch, count = valid_patch_mask.shape
    if batch < 1 or count < 2:
        raise ValueError("at least two patch positions are required")
    valid_count = valid_patch_mask.sum(dim=1)
    if torch.any(valid_count < 2).item():
        raise ValueError("each sample needs at least two valid patches")
    mask_count = torch.floor(valid_count.to(torch.float64) * mask_ratio).long()
    mask_count = torch.minimum(mask_count.clamp(min=1), valid_count - 1)
    if policy == "paired_symmetric":
        side = math.isqrt(count)
        if side * side != count:
            raise ValueError("paired_symmetric requires a square patch grid")
        valid_grid = valid_patch_mask.reshape(batch, side, side)
        if not torch.equal(valid_grid, valid_grid.transpose(1, 2)):
            raise ValueError("paired_symmetric requires a transpose-symmetric valid patch grid")
        diagonal = torch.arange(side, device=valid_patch_mask.device) * (side + 1)
        upper = torch.triu_indices(side, side, offset=1, device=valid_patch_mask.device)
        upper_flat = upper[0] * side + upper[1]
        lower_flat = upper[1] * side + upper[0]
        scores = torch.rand(
            (batch, count), device=valid_patch_mask.device, generator=generator
        )
        result = torch.zeros_like(valid_patch_mask)
        for index in range(batch):
            valid = valid_patch_mask[index]
            diagonal_indices = diagonal[valid[diagonal]]
            upper_selection = valid[upper_flat]
            upper_indices = upper_flat[upper_selection]
            lower_indices = lower_flat[upper_selection]
            target = int(mask_count[index].item())
            n_diagonal = diagonal_indices.numel()
            n_pairs = upper_indices.numel()
            feasible_diagonal = range(
                max(0, target - 2 * n_pairs), min(n_diagonal, target) + 1
            )
            choices = [d for d in feasible_diagonal if (target - d) % 2 == 0]
            if not choices:
                raise ValueError(
                    "mask ratio cannot leave both masked and visible patches "
                    "under paired_symmetric policy"
                )
            # Keep the diagonal masked at approximately the same rate as the
            # off-diagonal area, subject to an exact total and pair symmetry.
            expected_diagonal = target * n_diagonal / int(valid_count[index].item())
            diagonal_count = min(choices, key=lambda d: (abs(d - expected_diagonal), d))
            pair_count = (target - diagonal_count) // 2
            chosen_diagonal = diagonal_indices[
                scores[index, diagonal_indices].argsort()[:diagonal_count]
            ]
            chosen_pairs = scores[index, upper_indices].argsort()[:pair_count]
            result[index, chosen_diagonal] = True
            result[index, upper_indices[chosen_pairs]] = True
            result[index, lower_indices[chosen_pairs]] = True
        return result
    if policy != "independent":
        raise ValueError(f"unknown mask policy: {policy}")

    scores = torch.rand(
        (batch, count), device=valid_patch_mask.device, generator=generator
    ).masked_fill(~valid_patch_mask, torch.inf)
    ordering = scores.argsort(dim=1)
    in_sorted_order = torch.arange(count, device=scores.device)[None, :] < mask_count[:, None]
    result = torch.zeros_like(valid_patch_mask)
    return result.scatter_(dim=1, index=ordering, src=in_sorted_order)


class MaskedAutoencoder2D(nn.Module):
    """对 116×116 单通道矩阵做 MAE；forward 不接收疾病标签。

    PatchEmbed2D 与 encoder 可在后续分类模型中复用。mask token 在
    patch 嵌入后替换目标 token，编码器只见到可见值和位置占位符；
    decoder 从最深的 4×4 特征重建全部 1024 个 patch，但 loss 由外部
    `masked_mse` 仅在有效且被遮挡像素上计算。
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        patch_embed: PatchEmbed2D | None = None,
        decoder: MAEDecoder2D | None = None,
        in_channels: int = 1,
        embed_dim: int = 32,
        patch_size: int | tuple[int, int] = 4,
        original_size: tuple[int, int] = (116, 116),
        padded_size: tuple[int, int] = (128, 128),
        mask_ratio: float = 0.5,
        mask_policy: str = "independent",
    ) -> None:
        super().__init__()
        if not math.isfinite(mask_ratio) or not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio must be finite and strictly between 0 and 1")
        if mask_policy not in ("independent", "paired_symmetric"):
            raise ValueError("mask_policy must be independent or paired_symmetric")
        if in_channels < 1 or embed_dim < 1:
            raise ValueError("in_channels and embed_dim must be positive")
        self.patch_size = _pair(patch_size, "patch_size")
        self.original_size = _pair(original_size, "original_size")
        self.padded_size = _pair(padded_size, "padded_size")
        if any(padded < original for padded, original in zip(self.padded_size, self.original_size)):
            raise ValueError("padded_size must contain original_size")
        if any(size % patch for size, patch in zip(self.padded_size, self.patch_size)):
            raise ValueError("padded_size must be divisible by patch_size")
        patch_grid = tuple(size // patch for size, patch in zip(self.padded_size, self.patch_size))
        if any(size % 8 for size in patch_grid):
            raise ValueError("patch grid must be divisible by 8 for three 2× expansions")
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.mask_policy = mask_policy
        self.patch_grid = patch_grid
        if mask_policy == "paired_symmetric" and (
            self.patch_grid[0] != self.patch_grid[1]
            or self.original_size[0] != self.original_size[1]
            or self.patch_size[0] != self.patch_size[1]
        ):
            raise ValueError("paired_symmetric requires square image, patches, and patch grid")
        self.patch_embed = patch_embed if patch_embed is not None else PatchEmbed2D(
            in_channels=in_channels, embed_dim=embed_dim, patch_size=self.patch_size
        )
        if (
            self.patch_embed.in_channels != in_channels
            or self.patch_embed.embed_dim != embed_dim
            or self.patch_embed.patch_size != self.patch_size
        ):
            raise ValueError("patch_embed configuration does not match MAE configuration")
        self.encoder = encoder
        self.decoder = decoder if decoder is not None else MAEDecoder2D(
            input_grid=(patch_grid[0] // 8, patch_grid[1] // 8),
            patch_size=self.patch_size,
            out_channels=in_channels,
        )
        if (
            self.decoder.output_grid != patch_grid
            or self.decoder.patch_size != self.patch_size
            or self.decoder.out_channels != in_channels
        ):
            raise ValueError("decoder configuration does not match MAE configuration")
        self.mask_token = nn.Parameter(torch.empty(1, 1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        image: Tensor,
        *,
        patch_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        """[B,1,116,116] → 四项 [B,1024,16]/[B,1024] 重建输出。

        `patch_mask=True` 表示人为遮挡；补边块始终为 False。为复现
        验证/测试 mask，可显式传入固定的 bool [B,1024] patch_mask。
        """
        if image.ndim != 4 or image.shape[1] != self.in_channels:
            raise ValueError(
                f"expected image [B,{self.in_channels},H,W], got {tuple(image.shape)}"
            )
        padded, valid_pixels = pad_to_square(
            image, original_size=self.original_size, padded_size=self.padded_size
        )
        target_patches = patchify(padded, self.patch_size).detach()
        valid_patch_pixels = patchify(valid_pixels, self.patch_size)
        valid_patches = valid_patch_pixels.any(dim=-1)
        if patch_mask is None:
            patch_mask = sample_patch_mask(
                valid_patches, self.mask_ratio, generator=generator,
                policy=self.mask_policy,
            )
        else:
            if patch_mask.shape != valid_patches.shape or patch_mask.dtype != torch.bool:
                raise ValueError(
                    f"patch_mask must be bool {tuple(valid_patches.shape)}"
                )
            if patch_mask.device != image.device:
                raise ValueError("patch_mask and image must be on the same device")
            if torch.any(patch_mask & ~valid_patches).item():
                raise ValueError("patch_mask must not cover padding-only patches")
            if torch.any(patch_mask.sum(dim=1) < 1).item():
                raise ValueError("each sample needs at least one masked valid patch")
            if torch.any((valid_patches & ~patch_mask).sum(dim=1) < 1).item():
                raise ValueError("each sample needs at least one visible valid patch")
            if self.mask_policy == "paired_symmetric":
                mask_grid = patch_mask.reshape(image.shape[0], *self.patch_grid)
                if not torch.equal(mask_grid, mask_grid.transpose(1, 2)):
                    raise ValueError("paired_symmetric requires a transpose-symmetric patch_mask")

        tokens = self.patch_embed(padded)
        valid_grid = valid_patches.reshape(image.shape[0], *self.patch_grid, 1)
        masked_grid = patch_mask.reshape(image.shape[0], *self.patch_grid, 1)
        tokens = torch.where(valid_grid, tokens, torch.zeros_like(tokens))
        tokens = torch.where(masked_grid, self.mask_token.to(tokens.dtype), tokens)
        bottleneck = self.encoder(tokens)
        pred_patches = self.decoder(bottleneck)
        if pred_patches.shape != target_patches.shape:
            raise RuntimeError(
                f"decoder returned {tuple(pred_patches.shape)}, "
                f"expected {tuple(target_patches.shape)}"
            )
        return {
            "pred_patches": pred_patches,
            "target_patches": target_patches,
            "patch_mask": patch_mask,
            "valid_patch_pixels": valid_patch_pixels,
        }
