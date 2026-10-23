"""二维窗口 Mamba 块：行优先窗口序列、非循环移窗和任务提示。

生产路径只使用 ``mamba_ssm.Mamba``。``mamba_factory`` 是显式注入的测试接口，
任何测试替身的结果都不能作为真实 Mamba 训练结果报告。
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from .patches import _pair
from .windows import (
    shifted_window_partition_2d,
    shifted_window_reverse_2d,
    window_partition_2d,
    window_reverse_2d,
)


MambaFactory = Callable[..., nn.Module]


def _real_mamba_factory() -> MambaFactory:
    """只在实例化真实编码器时加载 CUDA Mamba；缺失时明确报错。"""
    try:
        from mamba_ssm import Mamba
    except (ImportError, OSError) as exc:
        raise ImportError(
            "SWMEncoder2D requires the real mamba_ssm.Mamba for production. "
            "Install a CUDA/PyTorch-compatible mamba-ssm build on the training "
            "server. A mamba_factory may be injected only for explicitly "
            "labelled unit tests."
        ) from exc
    return Mamba


class WindowMambaBlock2D(nn.Module):
    """LN → 窗口 Mamba → 残差 → LN → MLP → 残差。

    输入/输出均为 ``[B,H,W,C]``。窗口按二维行优先展开为
    ``[B*n_windows,window_h*window_w,C]``，移窗使用非循环补边；补边 token
    仍可能影响 Mamba 的内部状态，不能解释为完整的状态掩码。
    """

    def __init__(
        self,
        dim: int,
        window_size: int | tuple[int, int],
        *,
        shift_size: int | tuple[int, int] = 0,
        mlp_ratio: float = 4.0,
        prompt_length: int = 0,
        mamba_factory: MambaFactory | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(dim, int) or dim < 1:
            raise ValueError("dim must be a positive int")
        self.window_size = _pair(window_size, "window_size")
        if isinstance(shift_size, int):
            shift = (shift_size, shift_size)
        elif isinstance(shift_size, tuple) and len(shift_size) == 2 and all(
            isinstance(v, int) for v in shift_size
        ):
            shift = shift_size
        else:
            raise ValueError("shift_size must be a nonnegative int or pair")
        if any(s < 0 or s >= w for s, w in zip(shift, self.window_size)):
            raise ValueError("each shift must satisfy 0 <= shift < window")
        if not isinstance(mlp_ratio, (int, float)) or mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not isinstance(prompt_length, int) or prompt_length < 0:
            raise ValueError("prompt_length must be a nonnegative int")

        self.dim = dim
        self.shift_size = shift
        self.prompt_length = 0
        self.norm1 = nn.LayerNorm(dim)
        factory = mamba_factory if mamba_factory is not None else _real_mamba_factory()
        self.mamba = factory(d_model=dim, d_state=16, d_conv=4, expand=2)
        if not isinstance(self.mamba, nn.Module):
            raise TypeError("mamba_factory must return a torch.nn.Module")
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim)
        )
        self.register_parameter("prompt", None)
        if prompt_length:
            self.add_task_prompt(prompt_length)

    def add_task_prompt(self, length: int) -> None:
        """为这一块新建一个 ``[1,length,C]`` 提示，所有窗口共享。"""
        if not isinstance(length, int) or length < 1:
            raise ValueError("prompt length must be a positive int")
        if self.prompt is not None:
            raise RuntimeError("task prompt already exists; refusing to overwrite it")
        reference = self.norm1.weight
        prompt = nn.Parameter(reference.new_empty(1, length, self.dim))
        nn.init.trunc_normal_(prompt, std=0.02)
        self.prompt = prompt
        self.prompt_length = length

    def forward(self, x: Tensor, *, use_prompts: bool = False) -> Tensor:
        """``[B,H,W,C] → [B,H,W,C]``，提示输出在还原网格前去除。"""
        if x.ndim != 4 or x.shape[-1] != self.dim:
            raise ValueError(
                f"expected [B,H,W,{self.dim}], got {tuple(x.shape)}"
            )
        batch, height, width, _ = x.shape
        if batch < 1 or height < 1 or width < 1:
            raise ValueError("batch and spatial dimensions must be positive")
        wh, ww = self.window_size
        if height < wh or width < ww:
            raise ValueError("window_size cannot exceed the input grid")
        if use_prompts and self.prompt is None:
            raise RuntimeError("use_prompts=True requires add_task_prompt first")

        shortcut = x
        normalized = self.norm1(x)
        shifted = any(self.shift_size)
        if shifted:
            windows, valid_tokens = shifted_window_partition_2d(
                normalized, self.window_size, self.shift_size
            )
        else:
            windows = window_partition_2d(normalized, self.window_size)

        if use_prompts:
            # dtype 转换仍保留 prompt 的梯度；提示放在单向 Mamba 的前端。
            prefix = self.prompt.to(dtype=windows.dtype).expand(windows.shape[0], -1, -1)
            sequence = torch.cat((prefix, windows), dim=1)
        else:
            sequence = windows
        output = self.mamba(sequence)
        if output.shape != sequence.shape:
            raise RuntimeError(
                f"Mamba must preserve [windows,tokens,channels], got {tuple(output.shape)}"
            )
        if use_prompts:
            output = output[:, self.prompt_length :, :]

        if shifted:
            # 置零的是补边输出；Mamba 内部先前的状态传播仍可能受补边影响。
            output = output.masked_fill(~valid_tokens.unsqueeze(-1), 0)
            restored = shifted_window_reverse_2d(
                output,
                batch_size=batch,
                image_size=(height, width),
                window_size=self.window_size,
                shift_size=self.shift_size,
            )
        else:
            restored = window_reverse_2d(
                output,
                batch_size=batch,
                image_size=(height, width),
                window_size=self.window_size,
            )
        x = shortcut + restored
        return x + self.mlp(self.norm2(x))
