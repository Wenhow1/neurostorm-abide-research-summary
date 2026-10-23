"""只在下方、右方补边；显式跟踪原始有效像素。"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def pad_to_square(
    image: Tensor,
    *,
    original_size: tuple[int, int] = (116, 116),
    padded_size: tuple[int, int] = (128, 128),
) -> tuple[Tensor, Tensor]:
    """输入 [B,C,H,W]；输出补边图 [B,C,Hp,Wp] 和 bool mask [B,1,Hp,Wp]。

    标准化应在调用本函数之前、仅针对真实数据拟合和应用。补充值 0
    只是计算占位，不能通过数值是否为 0 判断像素是否有效。
    """
    if image.ndim != 4:
        raise ValueError(f"image must be [B,C,H,W], got {tuple(image.shape)}")
    batch, channels, height, width = image.shape
    if batch < 1 or channels < 1 or (height, width) != original_size:
        raise ValueError(
            f"expected nonempty [B,C,{original_size[0]},{original_size[1]}], "
            f"got {tuple(image.shape)}"
        )
    pad_h = padded_size[0] - height
    pad_w = padded_size[1] - width
    if pad_h < 0 or pad_w < 0:
        raise ValueError("padded_size must contain original_size")

    padded = F.pad(image, (0, pad_w, 0, pad_h), value=0)
    valid = torch.zeros(
        (batch, 1, padded_size[0], padded_size[1]),
        dtype=torch.bool,
        device=image.device,
    )
    valid[:, :, :height, :width] = True
    return padded, valid


def crop_to_original(
    padded: Tensor, *, original_size: tuple[int, int] = (116, 116)
) -> Tensor:
    """输入 [...,Hp,Wp]，输出 [...,H,W]；保留原坐标，不插值。"""
    if padded.ndim < 2 or any(
        actual < expected
        for actual, expected in zip(padded.shape[-2:], original_size)
    ):
        raise ValueError("padded tensor is smaller than original_size")
    return padded[..., : original_size[0], : original_size[1]]
