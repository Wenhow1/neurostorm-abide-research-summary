# 历史E1模型和训练模式适配
# 作用：说明旧patch代码怎样载入被冻结的mechanisms来源，以及历史分类头/Prompt模式。
# 输入输出：旧patch最终8×8×288；历史平均读出288维。
# 参数与配置：该适配器被train_fold导入供历史接口使用；当前脑区模型由base_models和models装配。
# 注意：模型加载包含源文件hash检查；依赖实际协议路径，不是单文件模型定义。
# 来源模块：source/adapters/model_adapter.py
# 原始源码镜像：代码核验/原始源码/source/adapters/model_adapter.py
# 源文件SHA256：28f60e7a36c619de3f67cee7b4bf3be509dc3d7b99639d5f76f9aa1fa5d71494
# 完整性：本文件保留来源模块全部可执行语句，仅格式化并增加中文注释；Python AST逐项一致。
# 运行边界：这是保留原import名称的代码阅读版本。中文文件名不能替代原模块路径；运行需原数据、划分、权重、软件环境与原工程布局。未声称独立可运行。
# 副作用提醒：含main或顶层训练/提取语句的脚本若直接执行会读写文件或使用CUDA。本次整理未执行模型。

"""Fixed E1/D1/M1 adapter for the independent current-FC regularization batch.

The frozen mechanisms/base_code sources are imported without editing them.
No training, CUDA allocation, or source import happens merely on importing this
file. A synthetic mixer is an explicit CPU-test-only injection, never production.
"""

from __future__ import annotations

import hashlib
import importlib.util
import random
import sys
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from instrumentation import (
    capture_rng,
    gradient_statistics,
    regularization_inventory,
    restore_rng,
    state_tensor_hash,
)


FIXED_MODEL = {
    "encoder": "E1",
    "decoder": "D1",
    "mask": "M1",
    "patch": 4,
    "stride": 2,
    "ratio": 0.5,
    "mask_policy": "paired_symmetric",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _seed_all(value):
    random.seed(value)
    try:
        import numpy as np

        np.random.seed(value)
    except ImportError:
        pass
    torch.manual_seed(value)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(value)


def load_frozen_mechanisms(config=None):
    config = config or {}
    source_root = Path(__file__).resolve().parent.parent / "model_source"
    path = Path(
        config.get("mechanisms_path", source_root / "code_v1/mechanisms.py")
    ).resolve()
    base = Path(
        config.get("base_code_path", path.parent.parent / "base_code")
    ).resolve()
    if (
        not path.is_file()
        or not (base / "neurostorm_2d/models/overlap_mae.py").is_file()
    ):
        raise FileNotFoundError(
            "Byte-identical frozen mechanisms.py and base_code are required"
        )
    expected = config.get("mechanisms_sha256")
    digest = _sha(path)
    if expected is not None and digest != expected:
        raise ValueError("Frozen mechanisms SHA256 mismatch")
    existing = sys.modules.get("neurostorm_2d")
    if existing is not None and existing.__file__ is not None:
        if not Path(existing.__file__).resolve().is_relative_to(base):
            raise ValueError(
                "Another neurostorm_2d source tree is already imported in this process"
            )
    sys.path.insert(0, str(base))
    name = "_currentfc_frozen_mechanisms_" + digest[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[name]
            raise
    module = sys.modules[name]
    actual = Path(sys.modules["neurostorm_2d.models.overlap_mae"].__file__).resolve()
    if not actual.is_relative_to(base):
        raise ValueError(
            "Imported base_code provenance differs from requested frozen source"
        )
    return module, {
        "mechanisms_path": str(path),
        "mechanisms_sha256": digest,
        "base_code_path": str(base),
    }


@contextmanager
def _mixer_context(module, factory):
    if factory is None:
        yield
        return
    with ExitStack() as stack:
        # Constructors transiently build both the historical and official
        # encoder before replacing them; all three actual factory references
        # must therefore be explicitly patched for CPU fixtures.
        for target in (
            module,
            sys.modules["neurostorm_2d.models.mamba_block"],
            sys.modules["neurostorm_2d.models.overlap_backbone"],
        ):
            stack.enter_context(
                patch.object(target, "_real_mamba_factory", lambda: factory)
            )
        yield


def _insert_mlp_dropout(block, prefix, probability):
    original = list(block.mlp.named_children())
    if (
        [name for name, _ in original] != ["0", "1", "2"]
        or not isinstance(original[0][1], nn.Linear)
        or not isinstance(original[1][1], nn.GELU)
        or not isinstance(original[2][1], nn.Linear)
    ):
        raise ValueError(f"Unexpected frozen MLP layout: {prefix}")
    # Preserve the exact original .0/.2 parameter namespaces across all arms.
    block.mlp = nn.Sequential(
        OrderedDict(
            [
                ("0", original[0][1]),
                ("1", original[1][1]),
                ("drop_after_gelu", nn.Dropout(probability)),
                ("2", original[2][1]),
                ("drop_after_linear", nn.Dropout(probability)),
            ]
        )
    )
    return [prefix + ".mlp.drop_after_gelu", prefix + ".mlp.drop_after_linear"]


class SelectiveClassifier(nn.Module):
    """Same MAE/head namespace and initialization as the frozen Classifier."""

    def __init__(self, original, dropout_paths, probability):
        super().__init__()
        self.mae, self.head, self.mode = original.mae, original.head, original.mode
        self.feature_dropout = nn.Dropout(probability) if probability else nn.Identity()
        self._dropout_paths = tuple(dropout_paths)
        if probability:
            self._dropout_paths += ("feature_dropout",)
        self.train(original.training)

    def train(self, mode=True):
        super().train(mode)
        # Matches the frozen override, then selectively re-enables ONLY the
        # new modules. eval() is not no_grad(); prompt gradients still propagate.
        self.mae.eval()
        modules = dict(self.named_modules())
        for name in self._dropout_paths:
            modules[name].train(mode)
        return self

    def forward_features(self, image):
        return self.mae.encode_full(image, use_prompts=self.mode in {"T1", "T2"}).mean(
            (1, 2)
        )

    def forward(self, image):
        return self.head(self.feature_dropout(self.forward_features(image))).squeeze(-1)


def get_backbone(model):
    return model.mae if isinstance(model, SelectiveClassifier) else model


def _parameter_groups(model, mode):
    groups = {}
    for name, parameter in model.named_parameters():
        if mode != "MAE":
            if name.split(".")[-1] == "prompt":
                group = "prompts"
            elif name.startswith("mae.encoder.stages.3.") and parameter.requires_grad:
                group = "encoder_last_stage_nonprompt"
            elif name.startswith("head."):
                group = "classification_head"
            else:
                group = "frozen_backbone"
        elif name.startswith("encoder.stages."):
            group = "encoder_stage_" + name.split(".")[2]
        elif name.startswith("encoder.pos_embeds."):
            group = "encoder_stage_" + name.split(".")[2]
        elif name.startswith("decoder.blocks."):
            group = "decoder_block_group_" + name.split(".")[2]
        elif name.startswith("decoder.first_patch_expanding."):
            group = "decoder_expansion_0"
        elif name.startswith("decoder.expands."):
            group = "decoder_expansion_" + str(int(name.split(".")[2]) + 1)
        elif name.startswith(("decoder.norm_up.", "decoder.decoder_pred.")):
            group = "decoder_prediction_head"
        elif (
            name.startswith(("value_projection.", "mask_projection."))
            or name == "mask_token"
        ):
            group = "embedding"
        else:
            raise ValueError(f"Unregistered parameter namespace: {name}")
        groups.setdefault(group, []).append(name)
    if sum(map(len, groups.values())) != len(list(model.named_parameters())):
        raise ValueError("Nonexclusive parameter partition")
    return groups


def make_model(
    task_mode,
    arm,
    seed,
    config=None,
    *,
    pretrained_state=None,
    mechanisms_module=None,
    _test_mixer_factory=None,
):
    """Return (model, info); seed is the effective fold/seed initialization seed.

    MAE uses seed; classification uses seed for MAE construction and seed+800000
    for the original Prompt/head constructor, preserving the historical path.
    Optional config initialization_seed/classification_seed overrides are logged.
    pretrained_state must be the raw MAE state_dict, loaded strictly BEFORE
    prompts/head are created. The worker must hash/bind its checkpoint source.
    """
    config = dict(config or {})
    mode = str(task_mode).upper()
    if mode not in {"MAE", "T0", "T1", "T2", "T2_NO_PROMPT"}:
        raise ValueError("task_mode must be MAE/T0/T1/T2/T2_NO_PROMPT")
    arm_id = arm["id"] if isinstance(arm, dict) else str(arm)
    if arm_id not in {"A" + str(i) for i in range(8)}:
        raise ValueError("Only predeclared A0-A7 arms are accepted")
    for key, expected in FIXED_MODEL.items():
        if key in config and config[key] != expected:
            raise ValueError(f"Fixed model configuration changed: {key}")
    rate = 0.1 if arm_id in {"A6", "A7"} and mode in {"MAE", "T2"} else 0.0
    if (
        isinstance(arm, dict)
        and "dropout" in arm
        and float(arm["dropout"]) != (0.1 if arm_id in {"A6", "A7"} else 0.0)
    ):
        raise ValueError("Arm Dropout probability differs from locked definition")
    if _test_mixer_factory is not None and not config.get("allow_test_mixer", False):
        raise ValueError("Synthetic mixer requires explicit allow_test_mixer=True")
    if mechanisms_module is None:
        module, source = load_frozen_mechanisms(config)
    else:
        if not config.get("allow_test_mixer", False):
            raise ValueError("Injected mechanisms module is test-only")
        module, source = mechanisms_module, {"mechanisms_path": "INJECTED_TEST_MODULE"}
    initial = int(config.get("initialization_seed", seed))
    classification_seed = int(config.get("classification_seed", seed + 800000))
    _seed_all(initial)
    with _mixer_context(module, _test_mixer_factory):
        mae = module.MechanismMAE(
            encoder="E1", decoder="D1", mask="M1", patch=4, stride=2
        )
    if mae.mask_ratio != 0.5 or mae.mask_policy != "paired_symmetric":
        raise ValueError("Unexpected MAE masking contract")
    if pretrained_state is not None:
        mae.load_state_dict(pretrained_state, strict=True)
    dropout_paths = []
    if mode == "MAE":
        model = mae
        if rate:
            for group, blocks in enumerate(mae.decoder.blocks):
                for block, layer in enumerate(blocks):
                    dropout_paths += _insert_mlp_dropout(
                        layer, f"decoder.blocks.{group}.{block}", rate
                    )
    else:
        _seed_all(classification_seed)
        original = module.Classifier(
            mae, "T1" if mode == "T0" else ("T2" if mode == "T2_NO_PROMPT" else mode)
        )
        if mode in {"T0", "T2_NO_PROMPT"}:
            for block in mae.encoder.iter_blocks():
                block.register_parameter("prompt", None)
            if mode == "T0":
                mae.requires_grad_(False)
            original.mode = mode
        if rate:
            for block, layer in enumerate(mae.encoder.stages[-1].blocks):
                dropout_paths += _insert_mlp_dropout(
                    layer, f"mae.encoder.stages.3.blocks.{block}", rate
                )
        model = SelectiveClassifier(original, dropout_paths, rate)
        dropout_paths = list(model._dropout_paths)
    groups = _parameter_groups(model, mode)
    inventory = regularization_inventory(model)
    info = {
        "task_mode": mode,
        "arm": arm_id,
        "fixed_model": FIXED_MODEL.copy(),
        "initialization_seed": initial,
        "classification_seed": classification_seed if mode != "MAE" else None,
        "source": source,
        "pretrained_state_loaded": pretrained_state is not None,
        "synthetic_mixer": _test_mixer_factory is not None
        or mechanisms_module is not None,
        "dropout_probability": rate,
        "dropout_paths": dropout_paths,
        "regularization_inventory": inventory,
        "gradient_groups": groups,
        "parameter_namespace": [
            {
                "name": name,
                "shape": list(parameter.shape),
                "elements": parameter.numel(),
                "requires_grad": parameter.requires_grad,
                "group": next(key for key, names in groups.items() if name in names),
            }
            for name, parameter in model.named_parameters()
        ],
    }
    return model, info


def _shapes(value):
    if torch.is_tensor(value):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {key: _shapes(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_shapes(item) for item in value]
    return type(value).__name__


@contextmanager
def trace_module_calls(model):
    """Shape/call-count hook helper for an explicitly invoked real GPU G0 step."""
    records, handles = {}, []
    for name, module in model.named_modules():

        def hook(current, inputs, output, key=name):
            row = records.setdefault(key, {"class": type(current).__name__, "calls": 0})
            row.update(
                {
                    "calls": row["calls"] + 1,
                    "last_input": _shapes(inputs),
                    "last_output": _shapes(output),
                    "training": current.training,
                }
            )

        handles.append(module.register_forward_hook(hook))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


def g0_forward_backward_probe(model, info, images, *, labels=None, pixel_mask=None):
    """One caller-invoked engineering step; NO optimizer update or queue launch.

    Reports actual device and synthetic/real mixer status. Restores original
    modes, grads and global RNG. This helper alone is not a G0 release certificate.
    """
    modes = [(module, module.training) for module in model.modules()]
    grads = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    rng = capture_rng()
    before = state_tensor_hash(model.state_dict())
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        with trace_module_calls(model) as calls:
            if info["task_mode"] == "MAE":
                output = model(images, pixel_mask=pixel_mask)
                selected = output["score_mask"]
                difference = (output["prediction"] - output["target"]).square()
                loss = torch.stack(
                    [
                        difference[i].masked_select(selected[i]).mean()
                        for i in range(images.shape[0])
                    ]
                ).mean()
            else:
                if labels is None:
                    raise ValueError(
                        "Synthetic engineering classifier labels are required"
                    )
                output = model(images)
                loss = F.binary_cross_entropy_with_logits(output, labels.to(output))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite engineering data loss")
            loss.backward()
        stats = gradient_statistics(model, info["gradient_groups"])
        if (
            not stats["global"]["all_finite"]
            or stats["global"]["frozen_with_grad_tensors"]
        ):
            raise ValueError(
                "Nonfinite or frozen-parameter gradient in engineering probe"
            )
        named = dict(model.named_parameters())
        disconnected = [
            group
            for group, names in info["gradient_groups"].items()
            if any(named[name].requires_grad for name in names)
            and not stats["groups"][group]["with_grad_elements"]
        ]
        if disconnected:
            raise ValueError(
                f"Entire trainable groups disconnected from graph: {disconnected}"
            )
        if state_tensor_hash(model.state_dict()) != before:
            raise ValueError(
                "A forward/backward-only probe unexpectedly changed model state"
            )
        return {
            "kind": "ENGINEERING_FORWARD_BACKWARD_ONLY_NOT_RELEASE_CERTIFICATE",
            "synthetic_mixer": info["synthetic_mixer"],
            "device": str(images.device),
            "data_loss": float(loss.detach()),
            "output": _shapes(output),
            "module_calls": calls,
            "gradient_statistics": stats,
            "parameter_gradients": {
                name: {
                    "requires_grad": parameter.requires_grad,
                    "grad_present": parameter.grad is not None,
                    "grad_l2": (
                        float(parameter.grad.detach().float().norm())
                        if parameter.grad is not None
                        else None
                    ),
                }
                for name, parameter in model.named_parameters()
            },
            "state_unchanged": True,
        }
    finally:
        for name, parameter in model.named_parameters():
            parameter.grad = grads[name]
        for module, mode in modes:
            module.training = mode
        restore_rng(rng)
