"""No-padding SWM backbone and geometry-matched decoder for overlapping patches.

Ragged boundary windows are batched by their real sequence length. No padding
token enters Mamba, and no cyclic roll joins opposite edges of the FC grid.
The production path always uses the existing real Mamba factory; an explicit
``mamba_factory`` is only a labelled unit-test injection point.
"""

from __future__ import annotations

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
        x.ndim != 4 or x.shape[0] < 1 or tuple(x.shape[1:]) != (*shape, dim)
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
            not isinstance(shift, tuple) or len(shift) != 2
            or any(isinstance(s, bool) or not isinstance(s, int) for s in shift)
            or any(s < 0 or s >= w for s, w in zip(shift, self.window_size))
        ):
            raise ValueError("shift_size must satisfy 0 <= shift < window per axis")
        if (
            not isinstance(mlp_ratio, (int, float)) or not math.isfinite(mlp_ratio)
            or mlp_ratio <= 0 or int(dim * mlp_ratio) < 1
        ):
            raise ValueError("mlp_ratio must produce a positive finite hidden size")
        self.shift_size = shift
        groups = window_index_groups(self.input_shape, self.window_size, shift)
        flat_indices = torch.cat([indices.reshape(-1) for indices in groups])
        if not torch.equal(
            flat_indices.sort().values,
            torch.arange(self.input_shape[0] * self.input_shape[1]),
        ):
            raise RuntimeError("window groups must cover every grid position exactly once")
        self._window_group_names = tuple(f"window_indices_{i}" for i in range(len(groups)))
        for name, indices in zip(self._window_group_names, groups):
            self.register_buffer(name, indices, persistent=False)

        self.norm1 = nn.LayerNorm(dim)
        factory = mamba_factory if mamba_factory is not None else _real_mamba_factory()
        self.mamba = factory(d_model=dim, d_state=16, d_conv=4, expand=2)
        if not isinstance(self.mamba, nn.Module):
            raise TypeError("mamba_factory must return a torch.nn.Module")
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
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
                mixed = mixed[:, self.prompt_length:, :]
            # Indices are disjoint between groups: no averaging, wrap, or padded
            # values are involved in reassembling a block's output.
            restored = restored.index_copy(
                1, indices.reshape(-1),
                mixed.to(dtype=restored.dtype).reshape(batch, n_windows * length, self.dim),
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
        self, *, dim: int, input_shape: tuple[int, int], depth: int,
        window_size: int | tuple[int, int], use_shift: bool, downsample: bool,
        mlp_ratio: float, mamba_factory: MambaFactory,
    ) -> None:
        super().__init__()
        window = _pair(window_size, "window_size")
        half_shift = tuple(w // 2 if size > w else 0 for size, w in zip(input_shape, window))
        self.blocks = nn.ModuleList(
            RaggedWindowMambaBlock2D(
                dim, input_shape, window, shift_size=half_shift if use_shift and i % 2 else 0,
                mlp_ratio=mlp_ratio, mamba_factory=mamba_factory,
            )
            for i in range(_positive_int(depth, "depth"))
        )
        self.downsample = OverlapPatchMerging2D(dim, input_shape) if downsample else None

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
                dim=self.stage_dims[i], input_shape=shape, depth=depths[i],
                window_size=self.window_sizes[i], use_shift=i < 2, downsample=i < 3,
                mlp_ratio=mlp_ratio, mamba_factory=factory,
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

    def __init__(self, in_dim: int, out_dim: int, output_shape: tuple[int, int]) -> None:
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
