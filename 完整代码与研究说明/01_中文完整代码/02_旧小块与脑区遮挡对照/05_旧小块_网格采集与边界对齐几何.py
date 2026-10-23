# 重叠patch及奇数网格的几何索引
# 作用：记录如何取4×4块，以及奇数维度合并时如何使用边界实际位置。
# 输入输出：116网格、4×4/stride2得到57×57；逐级合并时产生29、15、8网格。
# 参数与配置：真实位置索引与覆盖计数；不等同于可学习位置向量。
# 注意：
# 来源模块：source/neurostorm_2d/models/overlap_geometry.py
# 原始源码镜像：代码核验/原始源码/source/neurostorm_2d/models/overlap_geometry.py
# 源文件SHA256：e0e65556810bf9de3e43b48cbcd904c5bb4a69bb54e48470841705c6d33c9082
# 完整性：本文件保留来源模块全部可执行语句，仅格式化并增加中文注释；Python AST逐项一致。
# 运行边界：这是保留原import名称的代码阅读版本。中文文件名不能替代原模块路径；运行需原数据、划分、权重、软件环境与原工程布局。未声称独立可运行。
# 副作用提醒：含main或顶层训练/提取语句的脚本若直接执行会读写文件或使用CUDA。本次整理未执行模型。

"""Padding-free gather and average geometry for overlapping two-dimensional grids.

All indices refer to real input positions.  Odd merging grids keep their final
row and column by aligning the final group with the edge, which can overlap the
previous group.  ``average(gather(x))`` restores x; this geometric identity does
not imply that a learned dimensionality-reducing merge is invertible.
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor, nn


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _shape_pair(value: tuple[int, int], name: str) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be a pair of positive integers")
    return (_positive_int(value[0], name), _positive_int(value[1], name))


def _window_pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        side = _positive_int(value, name)
        return side, side
    return _shape_pair(value, name)


def _shift_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        pair = (value, value)
    elif isinstance(value, tuple) and len(value) == 2:
        pair = value
    else:
        raise ValueError("shift_size must be a nonnegative integer or pair")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in pair):
        raise ValueError("shift_size must be a nonnegative integer or pair")
    return pair


def _group_starts(size: int, kernel: int, stride: int, edge_align: bool) -> list[int]:
    if kernel > size:
        raise ValueError("kernel_size must not exceed either input dimension")
    starts = list(range(0, size - kernel + 1, stride))
    if edge_align and starts[-1] != size - kernel:
        starts.append(size - kernel)
    return starts


class GridGroups2D(nn.Module):
    """Gather real k×k groups and average their contributions onto a BHWC grid.

    ``indices`` is long [N,k²], with row-major groups and row-major positions
    within each group. ``coverage`` is floating [H,W] and strictly positive.
    Both are registered buffers, so callers should register this module as a
    child and move it to the same device as its tensors.
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        kernel_size: int,
        stride: int,
        edge_align: bool = True,
    ) -> None:
        super().__init__()
        self.input_shape = _shape_pair(input_shape, "input_shape")
        self.kernel_size = _positive_int(kernel_size, "kernel_size")
        self.stride = _positive_int(stride, "stride")
        if not isinstance(edge_align, bool):
            raise ValueError("edge_align must be bool")
        self.edge_align = edge_align
        height, width = self.input_shape
        row_starts = _group_starts(height, self.kernel_size, self.stride, edge_align)
        col_starts = _group_starts(width, self.kernel_size, self.stride, edge_align)
        self.output_shape = (len(row_starts), len(col_starts))
        self.area = self.kernel_size * self.kernel_size

        offset = torch.arange(self.kernel_size, dtype=torch.long, device="cpu")
        rows = torch.tensor(row_starts, dtype=torch.long, device="cpu")[
            :, None, None, None
        ]
        rows = rows + offset[None, None, :, None]
        cols = torch.tensor(col_starts, dtype=torch.long, device="cpu")[
            None, :, None, None
        ]
        cols = cols + offset[None, None, None, :]
        indices = (rows * width + cols).reshape(-1, self.area)
        coverage = torch.bincount(indices.reshape(-1), minlength=height * width)
        coverage = coverage.reshape(height, width).to(torch.float32)
        if torch.any(coverage == 0).item():
            raise ValueError(
                "group geometry leaves input positions uncovered; adjust stride "
                "or enable edge_align instead of padding or discarding positions"
            )
        self.register_buffer("indices", indices)
        self.register_buffer("coverage", coverage)

    def gather(self, grid: Tensor) -> Tensor:
        """[B,H,W,C] → [B,N,k²,C], including boolean grids used for masks."""
        if (
            grid.ndim != 4
            or tuple(grid.shape[1:3]) != self.input_shape
            or grid.shape[0] < 1
            or grid.shape[-1] < 1
        ):
            raise ValueError(
                f"grid must be nonempty [B,{self.input_shape[0]},{self.input_shape[1]},C]"
            )
        if grid.device != self.indices.device:
            raise ValueError("grid and geometry buffers must share a device")
        batch, height, width, channels = grid.shape
        selected = grid.reshape(batch, height * width, channels).index_select(
            1, self.indices.reshape(-1)
        )
        return selected.reshape(batch, self.indices.shape[0], self.area, channels)

    def average(self, values: Tensor) -> Tensor:
        """[B,N,k²,C] → [B,H,W,C], averaging every real position's coverage.

        Scatter addition and division remain in the autograd graph.  Values at
        overlapping group positions receive gradients scaled by 1/coverage.
        """
        if (
            values.ndim != 4
            or tuple(values.shape[1:3]) != (self.indices.shape[0], self.area)
            or values.shape[0] < 1
            or values.shape[-1] < 1
            or not torch.is_floating_point(values)
        ):
            raise ValueError(
                f"values must be floating and nonempty [B,{self.indices.shape[0]},{self.area},C]"
            )
        if values.device != self.indices.device:
            raise ValueError("values and geometry buffers must share a device")
        batch, _, _, channels = values.shape
        height, width = self.input_shape
        # Accumulate low-precision contributions in float32 before dividing.
        compute_dtype = (
            torch.float64 if values.dtype == torch.float64 else torch.float32
        )
        restored = torch.zeros(
            (batch, height * width, channels), dtype=compute_dtype, device=values.device
        )
        restored.index_add_(
            1,
            self.indices.reshape(-1),
            values.reshape(batch, -1, channels).to(compute_dtype),
        )
        restored = restored / self.coverage.reshape(1, height * width, 1).to(
            compute_dtype
        )
        return restored.reshape(batch, height, width, channels).to(values.dtype)


def _window_segments(size: int, window: int, shift: int) -> list[tuple[int, int]]:
    """Clip nominal intervals starting at -shift; never wrap an edge."""
    return [
        (max(start, 0), min(start + window, size))
        for start in range(-shift, size, window)
        if max(start, 0) < min(start + window, size)
    ]


def window_index_groups(
    shape: tuple[int, int],
    window_size: int | tuple[int, int],
    shift_size: int | tuple[int, int],
) -> list[Tensor]:
    """Return length-batched, padding-free rectangular window indices on CPU.

    Each result is long [n_windows,L].  Tokens inside each window are row-major;
    windows with the same L share a batch even if their rectangle orientations
    differ. Buckets are returned in descending L, with row-major window order
    inside each bucket. Every true grid position occurs exactly once overall.

    A positive shift moves nominal starts to -shift, retaining the alignment of
    the former noncyclic shifted-window implementation without adding tokens.
    Window sizes may exceed the grid; clipping still produces valid windows.
    """
    height, width = _shape_pair(shape, "shape")
    wh, ww = _window_pair(window_size, "window_size")
    sh, sw = _shift_pair(shift_size)
    if sh >= wh or sw >= ww:
        raise ValueError("each shift must be smaller than its window dimension")
    row_segments = _window_segments(height, wh, sh)
    col_segments = _window_segments(width, ww, sw)
    buckets: dict[int, list[Tensor]] = defaultdict(list)
    for top, bottom in row_segments:
        for left, right in col_segments:
            rows = torch.arange(top, bottom, dtype=torch.long, device="cpu")
            cols = torch.arange(left, right, dtype=torch.long, device="cpu")
            indices = (rows[:, None] * width + cols[None, :]).reshape(-1)
            buckets[indices.numel()].append(indices)
    return [torch.stack(buckets[length]) for length in sorted(buckets, reverse=True)]
