"""BHWC 网格的二维窗口拆分、还原与循环移窗边界掩码。"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .patches import _pair


def _check_window_shape(
    image_size: tuple[int, int], window_size: tuple[int, int]
) -> tuple[int, int]:
    height, width = _pair(image_size, "image_size")
    wh, ww = _pair(window_size, "window_size")
    if height % wh or width % ww:
        raise ValueError(
            f"image_size {(height, width)} must be divisible by window_size {(wh, ww)}"
        )
    return height // wh, width // ww


def window_partition_2d(
    image: Tensor, window_size: int | tuple[int, int]
) -> Tensor:
    """[B,H,W,C] → [B*(H/wh)*(W/ww),wh*ww,C]，窗口按行优先。"""
    wh, ww = _pair(window_size, "window_size")
    if image.ndim != 4:
        raise ValueError(f"image must be [B,H,W,C], got {tuple(image.shape)}")
    batch, height, width, channels = image.shape
    if min(batch, channels) < 1:
        raise ValueError("batch and channel dimensions must be positive")
    grid_h, grid_w = _check_window_shape((height, width), (wh, ww))
    return (
        image.reshape(batch, grid_h, wh, grid_w, ww, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch * grid_h * grid_w, wh * ww, channels)
    )


def window_reverse_2d(
    windows: Tensor,
    *,
    batch_size: int,
    image_size: tuple[int, int],
    window_size: int | tuple[int, int],
) -> Tensor:
    """[B*nw,M,C] → [B,H,W,C]，精确逆转 window_partition_2d。"""
    wh, ww = _pair(window_size, "window_size")
    height, width = _pair(image_size, "image_size")
    grid_h, grid_w = _check_window_shape((height, width), (wh, ww))
    if windows.ndim != 3:
        raise ValueError(f"windows must be [B*nw,M,C], got {tuple(windows.shape)}")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive int")
    count, tokens, channels = windows.shape
    if count != batch_size * grid_h * grid_w or tokens != wh * ww or channels < 1:
        raise ValueError(
            f"expected windows [{batch_size * grid_h * grid_w},{wh * ww},C], "
            f"got {tuple(windows.shape)}"
        )
    return (
        windows.reshape(batch_size, grid_h, grid_w, wh, ww, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch_size, height, width, channels)
    )


def _shifted_padding(
    image_size: tuple[int, int],
    window_size: int | tuple[int, int],
    shift_size: int | tuple[int, int],
) -> tuple[int, int, int, int]:
    height, width = _pair(image_size, "image_size")
    wh, ww = _pair(window_size, "window_size")
    sh, sw = _pair_allow_zero(shift_size, "shift_size")
    if sh >= wh or sw >= ww:
        raise ValueError("each shift must be smaller than its window dimension")
    # 上/左补 sh/sw；下/右补足窗口整数倍，避免把另一端循环接回来。
    top, left = sh, sw
    bottom = (-(top + height)) % wh
    right = (-(left + width)) % ww
    return left, right, top, bottom


def shifted_window_partition_2d(
    image: Tensor,
    window_size: int | tuple[int, int],
    shift_size: int | tuple[int, int],
) -> tuple[Tensor, Tensor]:
    """非循环移窗：[B,H,W,C] → ([B*nw,M,C], [B*nw,M] bool)。

    把原网格放在上/左补出的空位之后，再补齐下/右为窗口整数倍。
    ``valid_tokens`` 标记原图位置；这不是 Mamba 的状态掩码，但可用于
    检查、输出置零或后续有效特征汇总。若 H=W=32、窗口 8、移位 4，
    补后是 40×40，输出 25B 个各含 64 个 token 的窗口。
    """
    if image.ndim != 4:
        raise ValueError(f"image must be [B,H,W,C], got {tuple(image.shape)}")
    batch, height, width, channels = image.shape
    if min(batch, height, width, channels) < 1:
        raise ValueError("image dimensions must be positive")
    left, right, top, bottom = _shifted_padding(
        (height, width), window_size, shift_size
    )
    # F.pad 作用在 BCHW 最后两轴；显式换轴以免误补通道维。
    padded = F.pad(image.permute(0, 3, 1, 2), (left, right, top, bottom))
    padded = padded.permute(0, 2, 3, 1)
    valid = torch.ones((batch, 1, height, width), dtype=torch.bool, device=image.device)
    valid = F.pad(valid, (left, right, top, bottom)).permute(0, 2, 3, 1)
    return (
        window_partition_2d(padded, window_size),
        window_partition_2d(valid, window_size).squeeze(-1),
    )


def shifted_window_reverse_2d(
    windows: Tensor,
    *,
    batch_size: int,
    image_size: tuple[int, int],
    window_size: int | tuple[int, int],
    shift_size: int | tuple[int, int],
) -> Tensor:
    """非循环移窗逆变换：[B*nw,M,C] → 原坐标的 [B,H,W,C]。"""
    height, width = _pair(image_size, "image_size")
    left, right, top, bottom = _shifted_padding(
        (height, width), window_size, shift_size
    )
    padded = window_reverse_2d(
        windows,
        batch_size=batch_size,
        image_size=(top + height + bottom, left + width + right),
        window_size=window_size,
    )
    return padded[:, top : top + height, left : left + width, :]


def shifted_window_mask_2d(
    image_size: tuple[int, int],
    window_size: int | tuple[int, int],
    shift_size: int | tuple[int, int],
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """返回 [nw,M,M] bool：True 表示循环左上移位后两 token 可同窗交互。

    先以 ``torch.roll(x, shifts=(-sh,-sw), dims=(1,2))`` 移动 BHWC 网格，
    再分窗。掩码只阻止图像边缘循环回绕的 token 相互串通；内部跨原窗口
    的 token 仍可交互。它是成对边界标记，不能直接当作 Mamba 状态掩码。
    若窗口覆盖整轴，则该轴必须不移位。
    """
    height, width = _pair(image_size, "image_size")
    wh, ww = _pair(window_size, "window_size")
    sh, sw = _pair_allow_zero(shift_size, "shift_size")
    grid_h, grid_w = _check_window_shape((height, width), (wh, ww))
    if sh >= wh or sw >= ww:
        raise ValueError("each shift must be smaller than its window dimension")
    if (grid_h == 1 and sh != 0) or (grid_w == 1 and sw != 0):
        raise ValueError("an axis covered by one window must have zero shift")

    # 只需区分循环回绕前/后的区域；同一区域的 token 可以相互交互。
    row_wrap = (torch.arange(height, device=device) + sh >= height).long()
    col_wrap = (torch.arange(width, device=device) + sw >= width).long()
    region = 2 * row_wrap[:, None] + col_wrap[None, :]
    ids = window_partition_2d(region[None, :, :, None], (wh, ww)).squeeze(-1)
    return ids[:, :, None].eq(ids[:, None, :])


def _pair_allow_zero(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        result = (value, value)
    elif isinstance(value, tuple) and len(value) == 2 and all(
        isinstance(item, int) for item in value
    ):
        result = value
    else:
        raise ValueError(f"{name} must be a nonnegative int or pair")
    if result[0] < 0 or result[1] < 0:
        raise ValueError(f"{name} must be nonnegative, got {result}")
    return result
