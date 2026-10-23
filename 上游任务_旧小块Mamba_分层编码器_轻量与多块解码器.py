"""
旧小块Mamba路线。FC116×116先遮挡，再提取4×4重叠小块，16值投影到36通道。
主编码器通道36/72/144/288，含位置向量、窗口扫描、补边与移窗；末端8×8×288。
轻量B把旧解码器首组6块裁成2块，不能和脑区模型的单Linear轻量头混为一谈。
历史加宽W起始48、末端384；加深L在第三阶段加2块。坐标、下采样与增强是独立历史对照。
阅读顺序：U1网格→U2基础模块→U3输入遮挡→U4主架构→U5装配变化→U6/U7。
对应短报告01至04、08。历史NIAK论文4D模型未被伪装成当前静态FC实现。

用途：模型与任务设计阅读，不是独立运行入口。
本文件按章节汇集真实来源模块的完整类/函数；没有执行模型、训练或读取数据。
各章节仍保留各自的import名称与类名，同名类属于不同版本，不能顺序装配。
Mamba底层选择性状态空间算子来自mamba_ssm库；PyTorch/scikit-learn基础算子不在此重复实现。
完整可恢复工程、源文件位置和校验证据见私有完整归档。
"""

raise RuntimeError("这是代码阅读文件，请勿直接执行；运行使用原工程。")


# ========================================================================
# U1 网格采样与窗口边界
# 来源：05_旧小块_网格采集与边界对齐几何.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

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


# ========================================================================
# U2 空间合并 扩展与基础Mamba模块
# 来源：04_旧小块_空间合并扩展与基础Mamba模块.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

import math

from collections.abc import Iterator, Sequence

import torch

from torch import Tensor, nn

from .mamba_block import MambaFactory, _real_mamba_factory

from .overlap_geometry import GridGroups2D, window_index_groups

from .patches import _pair

def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int")
    return value

def _check_grid(x: Tensor, shape: tuple[int, int], dim: int, name: str) -> None:
    if (
        x.ndim != 4
        or x.shape[0] < 1
        or tuple(x.shape[1:]) != (*shape, dim)
        or not torch.is_floating_point(x)
    ):
        raise ValueError(f"{name} must be floating [B,{shape[0]},{shape[1]},{dim}]")

class RaggedWindowMambaBlock2D(nn.Module):
    """LN -> length-batched real windows -> Mamba -> residual -> LN/MLP.

    ``window_groups`` contains flattened row-major indices, grouped by window
    length. Within each block every grid position occurs exactly once. Prompts
    are shared across its windows and removed before scattering back to BHWC.
    """

    def __init__(
        self,
        dim: int,
        input_shape: tuple[int, int],
        window_size: int | tuple[int, int],
        *,
        shift_size: int | tuple[int, int] = 0,
        mlp_ratio: float = 4.0,
        mamba_factory: MambaFactory | None = None,
    ) -> None:
        super().__init__()
        self.dim = _positive_int(dim, "dim")
        self.input_shape = _pair(input_shape, "input_shape")
        self.window_size = _pair(window_size, "window_size")
        shift = (shift_size, shift_size) if isinstance(shift_size, int) else shift_size
        if (
            not isinstance(shift, tuple)
            or len(shift) != 2
            or any(isinstance(s, bool) or not isinstance(s, int) for s in shift)
            or any(s < 0 or s >= w for s, w in zip(shift, self.window_size))
        ):
            raise ValueError("shift_size must satisfy 0 <= shift < window per axis")
        if (
            not isinstance(mlp_ratio, (int, float))
            or not math.isfinite(mlp_ratio)
            or mlp_ratio <= 0
            or int(dim * mlp_ratio) < 1
        ):
            raise ValueError("mlp_ratio must produce a positive finite hidden size")
        self.shift_size = shift
        groups = window_index_groups(self.input_shape, self.window_size, shift)
        flat_indices = torch.cat([indices.reshape(-1) for indices in groups])
        if not torch.equal(
            flat_indices.sort().values,
            torch.arange(self.input_shape[0] * self.input_shape[1]),
        ):
            raise RuntimeError(
                "window groups must cover every grid position exactly once"
            )
        self._window_group_names = tuple(
            f"window_indices_{i}" for i in range(len(groups))
        )
        for name, indices in zip(self._window_group_names, groups):
            self.register_buffer(name, indices, persistent=False)

        self.norm1 = nn.LayerNorm(dim)
        factory = mamba_factory if mamba_factory is not None else _real_mamba_factory()
        self.mamba = factory(d_model=dim, d_state=16, d_conv=4, expand=2)
        if not isinstance(self.mamba, nn.Module):
            raise TypeError("mamba_factory must return a torch.nn.Module")
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.register_parameter("prompt", None)
        self.prompt_length = 0

    @property
    def window_groups(self) -> tuple[Tensor, ...]:
        return tuple(getattr(self, name) for name in self._window_group_names)

    def add_task_prompt(self, length: int) -> None:
        length = _positive_int(length, "prompt length")
        if self.prompt is not None:
            raise RuntimeError("task prompt already exists; refusing to overwrite it")
        self.prompt = nn.Parameter(self.norm1.weight.new_empty(1, length, self.dim))
        nn.init.trunc_normal_(self.prompt, std=0.02)
        self.prompt_length = length

    def forward(self, x: Tensor, *, use_prompts: bool = False) -> Tensor:
        _check_grid(x, self.input_shape, self.dim, "input")
        if use_prompts and self.prompt is None:
            raise RuntimeError("use_prompts=True requires add_task_prompt first")
        batch = x.shape[0]
        flat = self.norm1(x).reshape(batch, -1, self.dim)
        restored = torch.zeros_like(flat)
        for indices in self.window_groups:
            n_windows, length = indices.shape
            sequence = flat[:, indices.reshape(-1), :].reshape(
                batch * n_windows, length, self.dim
            )
            if use_prompts:
                prefix = self.prompt.to(dtype=sequence.dtype).expand(
                    batch * n_windows, -1, -1
                )
                sequence = torch.cat((prefix, sequence), dim=1)
            mixed = self.mamba(sequence)
            if mixed.shape != sequence.shape:
                raise RuntimeError("Mamba must preserve [windows,tokens,channels]")
            if use_prompts:
                mixed = mixed[:, self.prompt_length :, :]
            # Indices are disjoint between groups: no averaging, wrap, or padded
            # values are involved in reassembling a block's output.
            restored = restored.index_copy(
                1,
                indices.reshape(-1),
                mixed.to(dtype=restored.dtype).reshape(
                    batch, n_windows * length, self.dim
                ),
            )
        residual = x + restored.reshape_as(x)
        return residual + self.mlp(self.norm2(residual))

class OverlapPatchMerging2D(nn.Module):
    """Edge-aligned 2x2 groups, row-major concatenation, then 4C -> 2C.

    An odd final axis uses a real last 2-cell group anchored to the boundary.
    That group overlaps its predecessor by one cell; no zero cell is appended.
    """

    def __init__(self, dim: int, input_shape: tuple[int, int]) -> None:
        super().__init__()
        self.dim = _positive_int(dim, "dim")
        self.geometry = GridGroups2D(input_shape, 2, 2, edge_align=True)
        self.input_shape = self.geometry.input_shape
        self.output_shape = self.geometry.output_shape
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        _check_grid(x, self.input_shape, self.dim, "merge input")
        grouped = self.geometry.gather(x).flatten(start_dim=2)
        return self.reduction(self.norm(grouped)).reshape(
            x.shape[0], *self.output_shape, 2 * self.dim
        )

class OverlapSWMStage2D(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        input_shape: tuple[int, int],
        depth: int,
        window_size: int | tuple[int, int],
        use_shift: bool,
        downsample: bool,
        mlp_ratio: float,
        mamba_factory: MambaFactory,
    ) -> None:
        super().__init__()
        window = _pair(window_size, "window_size")
        half_shift = tuple(
            w // 2 if size > w else 0 for size, w in zip(input_shape, window)
        )
        self.blocks = nn.ModuleList(
            RaggedWindowMambaBlock2D(
                dim,
                input_shape,
                window,
                shift_size=half_shift if use_shift and i % 2 else 0,
                mlp_ratio=mlp_ratio,
                mamba_factory=mamba_factory,
            )
            for i in range(_positive_int(depth, "depth"))
        )
        self.downsample = (
            OverlapPatchMerging2D(dim, input_shape) if downsample else None
        )

    def forward(self, x: Tensor, *, use_prompts: bool = False) -> Tensor:
        for block in self.blocks:
            x = block(x, use_prompts=use_prompts)
        return self.downsample(x) if self.downsample is not None else x

class OverlapSWMEncoder2D(nn.Module):
    """Four real-grid stages; default 57 -> 29 -> 15 -> 8, C=16/32/64/128.

    A 115-grid becomes 115 -> 58 -> 29 -> 15. Only the first two stages
    alternate regular and half-shifted windows. Window boundaries are clipped
    to the real grid and grouped by length, never padded or cyclically rolled.
    """

    def __init__(
        self,
        embed_dim: int = 16,
        depths: Sequence[int] = (2, 2, 1, 1),
        window_sizes: Sequence[int | tuple[int, int]] = (8, 8, 8, 8),
        grid_size: int | tuple[int, int] = (57, 57),
        mlp_ratio: float = 4.0,
        mamba_factory: MambaFactory | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = _positive_int(embed_dim, "embed_dim")
        if len(depths) != 4 or len(window_sizes) != 4:
            raise ValueError("depths and window_sizes must contain four stages")
        for depth in depths:
            _positive_int(depth, "stage depth")
        self.grid_size = _pair(grid_size, "grid_size")
        self.window_sizes = tuple(_pair(w, "window_size") for w in window_sizes)
        self.stage_dims = tuple(embed_dim * (2**i) for i in range(4))
        factory = mamba_factory if mamba_factory is not None else _real_mamba_factory()
        stages, shapes = [], []
        shape = self.grid_size
        for i in range(4):
            shapes.append(shape)
            stage = OverlapSWMStage2D(
                dim=self.stage_dims[i],
                input_shape=shape,
                depth=depths[i],
                window_size=self.window_sizes[i],
                use_shift=i < 2,
                downsample=i < 3,
                mlp_ratio=mlp_ratio,
                mamba_factory=factory,
            )
            stages.append(stage)
            if stage.downsample is not None:
                shape = stage.downsample.output_shape
        self.stage_shapes = tuple(shapes)
        self.stages = nn.ModuleList(stages)
        self.pos_embed = nn.Parameter(torch.empty(1, *self.grid_size, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @property
    def merge_geometries(self) -> tuple[GridGroups2D, ...]:
        return tuple(stage.downsample.geometry for stage in self.stages[:-1])

    def iter_blocks(self) -> Iterator[RaggedWindowMambaBlock2D]:
        for stage in self.stages:
            yield from stage.blocks

    def add_task_prompts(self, length: int) -> None:
        length = _positive_int(length, "prompt length")
        if any(block.prompt is not None for block in self.iter_blocks()):
            raise RuntimeError("task prompts already exist")
        for block in self.iter_blocks():
            block.add_task_prompt(length)

    def freeze_backbone_for_prompt_tuning(self) -> None:
        blocks = tuple(self.iter_blocks())
        if not all(block.prompt is not None for block in blocks):
            raise RuntimeError("add_task_prompts must be called before freezing")
        self.requires_grad_(False)
        for block in blocks:
            block.prompt.requires_grad_(True)

    def forward(self, x: Tensor, *, use_prompts: bool = False) -> Tensor:
        _check_grid(x, self.grid_size, self.embed_dim, "encoder input")
        x = x + self.pos_embed.to(dtype=x.dtype)
        for stage in self.stages:
            x = stage(x, use_prompts=use_prompts)
        return x

class OverlapPatchExpanding2D(nn.Module):
    """Project each coarse cell into four fine cells and average overlap.

    The same edge-aligned geometry as forward merging determines every output
    position. This reverses its spatial mapping, not its learned lossy reduction.
    """

    def __init__(
        self, in_dim: int, out_dim: int, output_shape: tuple[int, int]
    ) -> None:
        super().__init__()
        self.in_dim = _positive_int(in_dim, "in_dim")
        self.out_dim = _positive_int(out_dim, "out_dim")
        self.geometry = GridGroups2D(output_shape, 2, 2, edge_align=True)
        self.input_shape = self.geometry.output_shape
        self.output_shape = self.geometry.input_shape
        self.proj = nn.Linear(in_dim, 4 * out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor) -> Tensor:
        _check_grid(x, self.input_shape, self.in_dim, "expand input")
        values = self.proj(x).reshape(x.shape[0], -1, 4, self.out_dim)
        return self.norm(self.geometry.average(values))

class OverlapMAEDecoder2D(nn.Module):
    """Reverse three saved stage geometries, then predict row-major patches.

    ``stage_shapes`` and ``stage_dims`` use encoder order (fine to coarse),
    unlike the legacy decoder. Output is [B,H0*W0,patch_area], without clipping,
    sigmoid, or skip connections from unmasked embeddings.
    """

    def __init__(
        self,
        stage_shapes: Sequence[tuple[int, int]],
        stage_dims: Sequence[int],
        patch_area: int,
    ) -> None:
        super().__init__()
        if len(stage_shapes) != 4 or len(stage_dims) != 4:
            raise ValueError("stage_shapes and stage_dims must contain four stages")
        self.stage_shapes = tuple(_pair(shape, "stage_shape") for shape in stage_shapes)
        self.stage_dims = tuple(_positive_int(dim, "stage_dim") for dim in stage_dims)
        self.patch_area = _positive_int(patch_area, "patch_area")
        self.input_grid = self.stage_shapes[-1]
        self.output_grid = self.stage_shapes[0]
        expansions = []
        for i in range(2, -1, -1):
            expansion = OverlapPatchExpanding2D(
                self.stage_dims[i + 1], self.stage_dims[i], self.stage_shapes[i]
            )
            if expansion.input_shape != self.stage_shapes[i + 1]:
                raise ValueError("stage_shapes do not follow edge-aligned 2x2 merging")
            expansions.append(expansion)
        self.expansions = nn.ModuleList(expansions)
        self.pixel_head = nn.Linear(self.stage_dims[0], patch_area)

    def forward(self, bottleneck: Tensor) -> Tensor:
        _check_grid(bottleneck, self.input_grid, self.stage_dims[-1], "bottleneck")
        grid = bottleneck
        for expansion in self.expansions:
            grid = expansion(grid)
        return self.pixel_head(grid).reshape(
            grid.shape[0], self.output_grid[0] * self.output_grid[1], self.patch_area
        )


# ========================================================================
# U3 遮挡 输入投影与重建映射
# 来源：03_方块遮挡与4x4重叠patch_输入投影和重建.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

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


# ========================================================================
# U4 E1分层编码器 D1解码器与旧分类模块
# 来源：01_旧小块Mamba_E1宽度36至288_D1解码器_完整机制.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

import sys, math

from pathlib import Path

import torch

from torch import nn

from torch.nn import functional as F

from neurostorm_2d.models.overlap_mae import (
    OverlapMaskedAutoencoder2D,
    overlap_masked_mse,
)

from neurostorm_2d.models.overlap_backbone import (
    OverlapPatchMerging2D,
    OverlapPatchExpanding2D,
)

from neurostorm_2d.models.mamba_block import _real_mamba_factory

class OfficialWindowBlock2D(nn.Module):
    """LN/pad/roll/window-Mamba/reverse/crop/residual/LN-MLP-residual.

    Official SwinTransformerBlock4D.forward_part1 does not use attn_mask
    in the Mamba branch. That behavior is preserved, including cyclic wrap.
    No STRD: selected official configuration defaults use_strd=False.
    """

    def __init__(self, dim, window=4, shift=0):
        super().__init__()
        self.dim = dim
        self.window = (window, window) if isinstance(window, int) else tuple(window)
        self.shift = (shift, shift) if isinstance(shift, int) else tuple(shift)
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = _real_mamba_factory()(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )
        self.register_parameter('prompt', None)

    def add_task_prompt(self, length):
        assert self.prompt is None
        self.prompt = nn.Parameter(self.norm1.weight.new_empty(1, length, self.dim))
        nn.init.trunc_normal_(self.prompt, std=0.02)

    def forward(self, x, *, use_prompts=False):
        b, h, w, c = x.shape
        wh, ww = min(h, self.window[0]), min(w, self.window[1])
        sh = self.shift[0] if h > self.window[0] else 0
        sw = self.shift[1] if w > self.window[1] else 0
        z = self.norm1(x)
        ph, pw = (-h) % wh, (-w) % ww
        z = F.pad(z, (0, 0, 0, pw, 0, ph))
        hp, wp = h + ph, w + pw
        if sh or sw:
            z = torch.roll(z, (-sh, -sw), (1, 2))
        z = (
            z.reshape(b, hp // wh, wh, wp // ww, ww, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, wh * ww, c)
        )
        if use_prompts:
            assert self.prompt is not None
            n = self.prompt.shape[1]
            z = torch.cat([self.prompt.expand(z.shape[0], -1, -1), z], dim=1)
            z = self.mamba(z)[:, n:]
        else:
            z = self.mamba(z)
        z = (
            z.reshape(b, hp // wh, wp // ww, wh, ww, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b, hp, wp, c)
        )
        if sh or sw:
            z = torch.roll(z, (sh, sw), (1, 2))
        z = x + z[:, :h, :w]
        return z + self.mlp(self.norm2(z))

class OfficialStage2D(nn.Module):
    def __init__(self, dim, shape, depth, window, downsample, full=False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                OfficialWindowBlock2D(
                    dim,
                    shape if full else window,
                    0 if full or i % 2 == 0 else window // 2,
                )
                for i in range(depth)
            ]
        )
        self.downsample = OverlapPatchMerging2D(dim, shape) if downsample else None

    def forward(self, x, *, use_prompts=False):
        for block in self.blocks:
            x = block(x, use_prompts=use_prompts)
        return self.downsample(x) if self.downsample is not None else x

class OfficialEncoder2D(nn.Module):
    """E1 selected YAML: width36, depths2/2/6/2, window4, last full.

    Retains edge-aligned 2D merging because 57/115 grids are odd. Temporal
    positions have no counterpart in FC; only spatial positions are adapted.
    """

    def __init__(self, shape):
        super().__init__()
        self.stage_dims = (36, 72, 144, 288)
        self.depths = (2, 2, 6, 2)
        self.stages = nn.ModuleList()
        self.pos_embeds = nn.ParameterList()
        shapes = []
        for i, (dim, dep) in enumerate(zip(self.stage_dims, self.depths)):
            shapes.append(shape)
            pos = nn.Parameter(torch.empty(1, *shape, dim))
            nn.init.trunc_normal_(pos, std=0.02)
            self.pos_embeds.append(pos)
            stage = OfficialStage2D(dim, shape, dep, 4, i < 3, full=i == 3)
            self.stages.append(stage)
            if stage.downsample is not None:
                shape = stage.downsample.output_shape
        self.stage_shapes = tuple(shapes)

    def iter_blocks(self):
        for s in self.stages:
            yield from s.blocks

    def add_task_prompts(self, n):
        for b in self.iter_blocks():
            b.add_task_prompt(n)

    def forward(self, x, *, use_prompts=False):
        for pos, stage in zip(self.pos_embeds, self.stages):
            x = stage(x + pos, use_prompts=use_prompts)
        return x

class OfficialExpand2D(OverlapPatchExpanding2D):
    def __init__(self, inc, outc, shape):
        super().__init__(inc, outc, shape)
        # Official expansion has no bias. In 2D, 4*(C/2)=2C outputs.
        self.proj = nn.Linear(inc, 4 * outc, bias=False)

class OfficialDecoder2D(nn.Module):
    """first expand -> block(s)/expand -> block(s)/expand -> block(s) -> LN/head.

    Official BasicLayerUp depths run encoder stages 2,1,0 in that order.
    Only real native grid cells are averaged when odd merge groups overlap.
    """

    def __init__(self, encoder, depths, area):
        super().__init__()
        ds = encoder.stage_dims
        ss = encoder.stage_shapes
        self.first_patch_expanding = OfficialExpand2D(ds[3], ds[2], ss[2])
        self.blocks = nn.ModuleList()
        self.expands = nn.ModuleList()
        for i in (2, 1, 0):
            self.blocks.append(
                nn.ModuleList(
                    [
                        OfficialWindowBlock2D(ds[i], 4, 0 if j % 2 == 0 else 2)
                        for j in range(depths[i])
                    ]
                )
            )
            if i > 0:
                self.expands.append(OfficialExpand2D(ds[i], ds[i - 1], ss[i - 1]))
        self.norm_up = nn.LayerNorm(ds[0])
        self.decoder_pred = nn.Linear(ds[0], area)

    def forward(self, x):
        x = self.first_patch_expanding(x)
        for i, blocks in enumerate(self.blocks):
            for b in blocks:
                x = b(x)
            if i < len(self.expands):
                x = self.expands[i](x)
        return self.decoder_pred(self.norm_up(x)).flatten(1, 2)

class MechanismMAE(OverlapMaskedAutoencoder2D):
    def __init__(self, encoder='E0', decoder='D0', mask='M0', patch=4, stride=2):
        self.encoder_kind, self.decoder_kind, self.mask_kind = encoder, decoder, mask
        dim = 16 if encoder == 'E0' else 36
        depths = (2, 2, 1, 1) if encoder == 'E0' else (2, 2, 6, 2)
        # The full constructor is deterministically seeded by fold and seed.
        # E1 replaces the initial legacy-shaped encoder; D1 replaces D0.
        # Pairwise A/B jobs load the exact same resulting MAE checkpoint.
        super().__init__(
            patch_size=patch,
            stride=stride,
            embed_dim=dim,
            depths=depths,
            window_sizes=(8, 8, 8, 8),
            mask_unit=4,
            mask_policy='paired_symmetric',
        )
        if encoder == 'E1':
            self.encoder = OfficialEncoder2D(self.geometry.output_shape)
        if decoder == 'D1':
            self.decoder = OfficialDecoder2D(self.encoder, depths, patch * patch)
        if mask == 'M1':
            self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.depths = depths

    def embedded_input(self, image, pixel_mask):
        tokens = super().embedded_input(image, pixel_mask)
        if self.mask_kind == 'M1':
            full = self.geometry.gather(pixel_mask.permute(0, 2, 3, 1)).all(dim=2)
            flat = tokens.flatten(1, 2)
            flat = torch.where(full, self.mask_token.to(flat.dtype), flat)
            tokens = flat.reshape_as(tokens)
        return tokens

class Classifier(nn.Module):
    def __init__(self, mae, mode):
        super().__init__()
        assert mode in ('T0', 'T1', 'T2')
        self.mae = mae
        self.mode = mode
        mae.requires_grad_(False)
        if mode != 'T0':
            mae.encoder.add_task_prompts(50)
            for b in mae.encoder.iter_blocks():
                b.prompt.requires_grad_(True)
        if mode == 'T2':
            mae.encoder.stages[-1].requires_grad_(True)
        self.head = nn.Linear(mae.encoder.stage_dims[-1], 1)

    def train(self, mode=True):
        super().train(mode)
        self.mae.eval()
        return self

    def forward(self, image):
        return self.head(
            self.mae.encode_full(image, use_prompts=self.mode != 'T0').mean((1, 2))
        ).squeeze(-1)


# ========================================================================
# U5 B轻量头 W加宽 L加深的实际装配
# 来源：07_旧小块Mamba_W加宽384与L加深14块_完整训练流程.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

def make(f, seed, arm):
    if seed == 1:
        m = old.make('MAE', f)
    else:
        m = old.make_model('MAE',
             'A2',
             1,
             dict(initialization_seed=2026102200 + f + 1000000 * (seed - 1),
             classification_seed=2026102200 + f + 800000,
             mechanisms_path=str(S / 'model_source/code_v1/mechanisms.py'),
             mechanisms_sha256=read(S / 'protocol.json')['mechanisms_sha256'],
             base_code_path=str(S)))[0]
    m.decoder.blocks[0] = torch.nn.ModuleList(list(m.decoder.blocks[0])[:2])
    if arm in ['W', 'L']:
        mod = sys.modules[type(m.encoder).__module__]
        shared = {k: v.clone() for k, v in m.state_dict().items()}
        with torch.random.fork_rng():
            torch.manual_seed(2026102200 + f + 1000000 * (seed - 1) + 7000000)
            if arm == 'L':
                for shift in [0, 2]:
                    m.encoder.stages[2].blocks.append(mod.OfficialWindowBlock2D(144, 4, shift))
                m.encoder.depths = (2, 2, 8, 2)
                m.depths = (2, 2, 8, 2)
            else:
                m.embed_dim = 48
                m.value_projection = torch.nn.Linear(16, 48)
                m.mask_projection = torch.nn.Linear(16, 48, bias=False)
                m.mask_token = torch.nn.Parameter(torch.zeros(1, 1, 48))
                enc = m.encoder
                enc.stage_dims = (48, 96, 192, 384)
                enc.stages = torch.nn.ModuleList()
                enc.pos_embeds = torch.nn.ParameterList()
                for i, (dim, dep, shape) in enumerate(zip(enc.stage_dims, enc.depths, enc.stage_shapes)):
                    pos = torch.nn.Parameter(torch.empty(1, *shape, dim))
                    torch.nn.init.trunc_normal_(pos, std=0.02)
                    enc.pos_embeds.append(pos)
                    enc.stages.append(mod.OfficialStage2D(dim, shape, dep, 4, i < 3, full=i == 3))
                m.decoder.first_patch_expanding = mod.OfficialExpand2D(384, 144, m.encoder.stage_shapes[2])
        new = m.state_dict()
        for k, v in shared.items():
            if k in new and new[k].shape == v.shape:
                new[k] = v
        m.load_state_dict(new)
    return m


# ========================================================================
# U6 S0与S1下采样编码器
# 来源：08_旧小块Mamba_S0先合并与S1延后合并_完整编码器.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

import torch

from torch import nn

from torch.nn import functional as F

class SameWidthMerge(nn.Module):

    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry
        self.input_shape = geometry.input_shape
        self.output_shape = geometry.output_shape
        self.norm = nn.LayerNorm(1152)
        self.reduction = nn.Linear(1152, 288, bias=False)

    def forward(self, x):
        assert tuple(x.shape[1:]) == (15, 15, 288)
        groups = self.geometry.gather(x).flatten(start_dim=2)
        return self.reduction(self.norm(groups)).reshape(x.shape[0], 8, 8, 288)

class OrderedEncoder(nn.Module):
    """Shared fine-grid position before both routes; only block/merge order differs."""

    def __init__(self, base, late):
        super().__init__()
        self.late = late
        self.stage_dims = base.stage_dims
        self.depths = base.depths
        self.stage_shapes = base.stage_shapes
        self.stages = base.stages
        self.pos_embeds = base.pos_embeds
        geometry = self.stages[2].downsample.geometry
        self.stages[2].downsample = None
        self.bridge = nn.Linear(144, 288, bias=False)
        self.merge = SameWidthMerge(geometry)
        coarse = self.pos_embeds[3].detach().permute(0, 3, 1, 2)
        fine = F.interpolate(coarse,
             size=(15,
             15),
             mode='bilinear',
             align_corners=False).permute(0,
             2,
             3,
             1).contiguous()
        self.pos_embeds[3] = nn.Parameter(fine)
        for b in self.stages[3].blocks:
            b.window = (15, 15)
            b.shift = (0, 0)

    def iter_blocks(self):
        for s in self.stages:
            yield from s.blocks

    def forward(self, x, *, use_prompts=False):
        for i in range(3):
            x = self.stages[i](x + self.pos_embeds[i], use_prompts=use_prompts)
        x = self.bridge(x) + self.pos_embeds[3]
        if self.late:
            x = self.stages[3](x, use_prompts=use_prompts)
            x = self.merge(x)
        else:
            x = self.merge(x)
            x = self.stages[3](x, use_prompts=use_prompts)
        assert tuple(x.shape[1:]) == (8, 8, 288)
        return x


# ========================================================================
# U7 真实或打乱坐标分支
# 来源：10_旧小块Mamba_真实或打乱坐标_完整坐标分支.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

import torch, numpy as np

from torch import nn

class CoordinateEncoder(nn.Module):

    def __init__(self, base, arm, atlas):
        super().__init__()
        self.stages = base.stages
        self.pos_embeds = base.pos_embeds
        self.stage_dims = base.stage_dims
        self.stage_shapes = base.stage_shapes
        self.depths = base.depths
        xyz = torch.from_numpy(np.load(atlas / 'coordinates_mm.npy')).float() / 100.0
        if arm == 'P2':
            xyz = xyz[torch.from_numpy(np.load(atlas / 'permutation.npy'))]
        delta = xyz[:, None, :] - xyz[None, :, :]
        pair = torch.cat([(xyz[:,
             None,
             :] + xyz[None,
             :,
             :]) / 2,
             delta.abs(),
             delta.norm(dim=-1,
             keepdim=True)],
             dim=-1)
        self.register_buffer('geo_pair', pair[None])
        self.geo_mlp = nn.Sequential(nn.Linear(7, 36), nn.GELU(), nn.Linear(36, 36))
        self.geo_projections = nn.ModuleList([nn.Identity()] + [nn.Linear(36,
             d,
             bias=False) for d in self.stage_dims[1:]])
        self.geo_gates = nn.Parameter(torch.zeros(4))
        from neurostorm_2d.models.overlap_geometry import GridGroups2D
        self.geo_patch = GridGroups2D((116, 116), 4, 2, edge_align=False)

    def iter_blocks(self):
        for s in self.stages:
            yield from s.blocks

    def forward(self, x, *, use_prompts=False):
        g = self.geo_mlp(self.geo_pair)
        g = self.geo_patch.gather(g).mean(2).reshape(1, 57, 57, 36)
        for i, (p, stage) in enumerate(zip(self.pos_embeds, self.stages)):
            x = stage(x + p + self.geo_gates[i] * self.geo_projections[i](g), use_prompts=use_prompts)
            if stage.downsample is not None:
                g = stage.downsample.geometry.gather(g).mean(2).reshape(1, *stage.downsample.output_shape, 36)
        return x


# ========================================================================
# U8 噪声与时序窗口增强
# 来源：09_旧小块Mamba_G1噪声与G2时序窗口_实际增强模块.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

from pathlib import Path

import numpy as np, hashlib, json

R = Path(__file__).resolve().parent

def rng_for(kind, fold, seed, epoch, sid):
    return np.random.default_rng(int(hashlib.sha256(f'augv1|{kind}|{fold}|{seed}|{epoch}|{sid}'.encode()).hexdigest()[:16],
         16))

def selected(f, s, e, sid):
    return rng_for('switch', f, s, e, sid).random() < 0.5

class Augment:

    def __init__(self, arm, fold, seed, ids, raw, norm):
        self.arm, self.fold, self.seed, self.ids, self.raw, self.norm = (arm, fold, seed, ids, raw, norm)
        self.cache = {}

    def arrays(self, q, epoch, mask):
        target = np.array(self.raw[q], copy=True)
        inputs = target.copy()
        changed = 0
        noise_sq = 0.0
        noise_n = 0
        clipped = 0
        window_starts = []
        for k, i in enumerate(q):
            sid = str(self.ids[i])
            if self.arm not in ['G1', 'G2'] or not selected(self.fold, self.seed, epoch, sid):
                continue
            changed += 1
            if self.arm == 'G1':
                u, v = np.triu_indices(116, 1)
                r = rng_for('noise', self.fold, self.seed, epoch, sid)
                eps = r.normal(0, 0.02, len(u))
                visible = ~mask[k, 0, u, v]
                u, v, eps = (u[visible], v[visible], eps[visible])
                a = inputs[k, u, v] + eps
                clipped += int(np.sum(np.abs(a) > 1))
                a = np.clip(a, -1, 1)
                actual = a - inputs[k, u, v]
                inputs[k, u, v] = a
                inputs[k, v, u] = a
                noise_sq += float(np.sum(actual ** 2))
                noise_n += len(a)
            else:
                if sid not in self.cache:
                    self.cache[sid] = np.load(R / 'windows' / f'{sid}.npy', mmap_mode='r')
                w = self.cache[sid]
                start = int(rng_for('window', self.fold, self.seed, epoch, sid).integers(len(w)))
                target[k] = w[start]
                inputs[k] = target[k]
                window_starts.append([sid, start])
        return (inputs,
             target,
             {'augmented': changed,
             'subjects': len(q),
             'noise_squared_sum': noise_sq,
             'noise_edges': noise_n,
             'noise_clipped_edges': clipped,
             'windows': window_starts})
