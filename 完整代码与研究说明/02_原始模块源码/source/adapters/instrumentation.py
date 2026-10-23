"""G0 engineering utilities; NOT integrated with any training queue.

Call gradient_statistics only after accumulation and AMP unscale, before/after
clipping. These functions never unscale, clip, zero, or otherwise change grads.
Call explicit penalties once per optimizer step. Checkpoints are supported only
after a completed/skipped optimizer step and zero_grad(set_to_none=True).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from contextlib import contextmanager
from typing import Mapping, Sequence

import torch
from torch import nn

try:
    import numpy as np
except ImportError:
    np = None


_CONV = (nn.Conv1d, nn.Conv2d, nn.Conv3d,
         nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
_MAMBA_CLASSES = frozenset({"Mamba", "Mamba2", "Mamba2Simple"})


def _under(name: str, root: str) -> bool:
    return name == root or name.startswith(root + ".")


def _parameters(model: nn.Module) -> dict:
    # Aliases make per-name eligibility and disjoint accounting ambiguous.
    pairs = list(model.named_parameters(remove_duplicate=False))
    if len({id(p) for _, p in pairs}) != len(pairs):
        raise ValueError("Tied/aliased parameters require an explicit canonical-name policy")
    return dict(pairs)


def regularization_inventory(
    model: nn.Module, *, mamba_roots: Sequence[str] = (),
    mamba_types: tuple = (), component_roots: Mapping[str, Sequence[str]] | None = None,
) -> dict:
    """List every parameter, exclusion reason, and element-level coverage.

    Actual mechanisms.py blocks use a child named ``mamba``. That exact path
    segment and known Mamba class names identify whole subtrees, not just their
    projection weights. Additional implementation classes/roots must be passed
    explicitly; an unfamiliar implementation is never inferred from tensor size.
    """
    modules = dict(model.named_modules())
    if any(not root or root not in modules for root in mamba_roots):
        raise ValueError("Every explicit Mamba root must name an existing non-root module")
    roots = set(mamba_roots)
    for name, module in modules.items():
        if (name.split(".")[-1] == "mamba"
                or type(module).__name__ in _MAMBA_CLASSES
                or (mamba_types and isinstance(module, mamba_types))):
            if not name:
                raise ValueError("The entire model is Mamba; pass a containing model")
            roots.add(name)
    component_roots = component_roots or {
        "encoder": ("encoder", "mae.encoder"),
        "decoder": ("decoder", "mae.decoder"), "head": ("head",),
    }
    coverage = {key: {"trainable_elements": 0, "regularized_elements": 0}
                for key in [*component_roots, "other"]}
    rows = []
    for name, parameter in _parameters(model).items():
        matches = [group for group, prefixes in component_roots.items()
                   if any(_under(name, prefix) for prefix in prefixes)]
        if len(matches) > 1:
            raise ValueError(f"Overlapping component roots: {name}: {matches}")
        group = matches[0] if matches else "other"
        owner_path, _, leaf = name.rpartition(".")
        owner = modules[owner_path]
        semantic_segments = name.lower().split(".")
        special = any(segment in {"prompt", "mask_token", "pos_embeds", "pos_embed",
                                  "position_embeddings", "position_embedding", "position_encoding"}
                      for segment in semantic_segments)
        if not parameter.requires_grad:
            reason = "frozen"
        elif any(_under(name, root) for root in roots):
            reason = "entire_mamba_subtree"
        elif special:
            reason = "normalization_prompt_position_or_mask_token"
        elif leaf != "weight":
            reason = "not_weight_bias_or_other_parameter"
        elif not isinstance(owner, (nn.Linear, *_CONV)):
            reason = "not_ordinary_linear_or_conv_weight"
        else:
            reason = "eligible"
        eligible = reason == "eligible"
        n = parameter.numel()
        if parameter.requires_grad:
            coverage[group]["trainable_elements"] += n
        if eligible:
            coverage[group]["regularized_elements"] += n
        rows.append({"name": name, "owner_class": type(owner).__name__,
                     "shape": list(parameter.shape), "elements": n,
                     "requires_grad": parameter.requires_grad, "component": group,
                     "regularized": eligible, "reason": reason})
    for item in coverage.values():
        total = item["trainable_elements"]
        item["coverage_fraction"] = item["regularized_elements"] / total if total else 0.0
    return {"schema": 1, "mamba_roots": sorted(roots), "parameters": rows,
            "coverage": coverage}


def regularization_terms(model: nn.Module, inventory: dict,
                         l1: float = 1e-7, l2: float = 1e-5) -> dict:
    """Explicit L1 and half-squared L2 over the audited eligible tensors."""
    if l1 < 0 or l2 < 0:
        raise ValueError("Penalty coefficients must be nonnegative")
    parameters = _parameters(model)
    rows = inventory["parameters"]
    if [r["name"] for r in rows] != list(parameters):
        raise ValueError("Inventory no longer describes this model")
    if any(r["requires_grad"] != parameters[r["name"]].requires_grad for r in rows):
        raise ValueError("Trainable/frozen configuration changed after inventory")
    chosen = [parameters[r["name"]] for r in rows if r["regularized"]]
    prototype = next(iter(parameters.values()), None)
    zero = prototype.new_zeros(()) if prototype is not None else torch.tensor(0.0)
    # FP32 reduction also avoids half-precision overflow under autocast.
    one = sum((p.float().abs().sum() for p in chosen), zero)
    two = sum((p.float().square().sum() for p in chosen), zero)
    return {"l1": one * l1, "l2": two * (0.5 * l2),
            "total": one * l1 + two * (0.5 * l2),
            "regularized_elements": sum(p.numel() for p in chosen)}


def groups_from_roots(model: nn.Module, roots: Mapping[str, Sequence[str]]) -> dict:
    """Build an exhaustive, mutually exclusive parameter partition; fail closed."""
    groups = {key: [] for key in roots}
    for name in _parameters(model):
        matches = [key for key, prefixes in roots.items()
                   if any(_under(name, root) for root in prefixes)]
        if len(matches) != 1:
            raise ValueError(f"Exactly one gradient group required for {name}: {matches}")
        groups[matches[0]].append(name)
    return groups


def _gradient_summary(items: Sequence[tuple]) -> dict:
    return _gradient_values_summary([(name, p.grad, p.requires_grad) for name, p in items])


def _gradient_values_summary(items: Sequence[tuple]) -> dict:
    summary = {"parameter_tensors": len(items), "none_tensors": 0,
               "with_grad_elements": 0, "zero_tensors": 0, "zero_elements": 0,
               "nonfinite_tensors": 0, "nonfinite_elements": 0,
               "frozen_with_grad_tensors": 0}
    square_sum = 0.0
    maximum = 0.0
    for _, grad, requires_grad in items:
        if grad is None:
            summary["none_tensors"] += 1
            continue
        if grad.is_sparse:
            raise ValueError("Sparse gradients need a separate audited accounting policy")
        values = grad.detach().to(dtype=torch.float64)
        finite = torch.isfinite(values)
        bad = int((~finite).sum().item())
        summary["with_grad_elements"] += values.numel()
        summary["nonfinite_elements"] += bad
        summary["nonfinite_tensors"] += int(bool(bad))
        summary["frozen_with_grad_tensors"] += int(not requires_grad)
        zeros = int((values == 0).sum().item())
        summary["zero_elements"] += zeros
        summary["zero_tensors"] += int(zeros == values.numel())
        square_sum += float(values.square().sum().item())
        if values.numel():
            maximum = max(maximum, float(values.abs().max().item()))
    good = not summary["nonfinite_elements"]
    n = summary["with_grad_elements"]
    summary.update({"all_finite": good,
                    "l2": math.sqrt(square_sum) if good else None,
                    "rms": math.sqrt(square_sum / n) if good and n else (0.0 if good else None),
                    "max_abs": maximum if good else None})
    return summary


def gradient_statistics(model: nn.Module, groups: Mapping[str, Sequence[str]]) -> dict:
    """Read-only post-unscale statistics, including an independent norm identity."""
    parameters = _parameters(model)
    flat = [name for names in groups.values() for name in names]
    if len(flat) != len(set(flat)) or set(flat) != set(parameters):
        raise ValueError("Gradient groups must cover all parameters exactly once")
    grouped = {key: _gradient_summary([(name, parameters[name]) for name in names])
               for key, names in groups.items()}
    overall = _gradient_summary(list(parameters.items()))
    combined = (math.sqrt(sum(item["l2"] ** 2 for item in grouped.values()))
                if overall["all_finite"] else None)
    identity = bool(overall["all_finite"] and math.isclose(
        overall["l2"], combined, rel_tol=1e-12, abs_tol=1e-12))
    return {"groups": grouped, "global": overall,
            "sqrt_group_squared_norms": combined, "norm_identity_pass": identity,
            "gradient_stage": "caller_must_record_post_unscale_pre_or_post_clip"}


def analytic_gradient_decomposition(
    model: nn.Module, groups: Mapping[str, Sequence[str]], inventory: dict,
    *, post_unscale_pre_clip: bool, l1: float = 1e-7, l2: float = 1e-5,
    regularizer_applications: int = 1,
) -> dict:
    """Infer data gradients from combined grads, without forward/backward/RNG.

    The caller certifies grads are post-accumulation, unscaled and pre-clip, and
    contain exactly one application of these explicit penalties. Data gradient
    is combined - lambda1*sign(weight) - lambda2*weight. AdamW decay is excluded.
    These are analytic diagnostics, not an exact floating-point reconstruction:
    FP32 accumulation/subtraction can lose small terms. None is kept as None;
    an eligible tensor with no grad is reported, never treated as evidence that
    its nonzero penalty participated in an update.
    """
    if not post_unscale_pre_clip or regularizer_applications != 1:
        raise ValueError("Requires post-unscale pre-clip grads and exactly one penalty application")
    if l1 < 0 or l2 < 0 or not math.isfinite(l1 + l2):
        raise ValueError("Finite nonnegative penalty coefficients required")
    parameters = _parameters(model)
    flat = [name for names in groups.values() for name in names]
    if len(flat) != len(set(flat)) or set(flat) != set(parameters):
        raise ValueError("Gradient groups must cover all parameters exactly once")
    rows = inventory["parameters"]
    if [row["name"] for row in rows] != list(parameters):
        raise ValueError("Inventory no longer describes this model")
    if any(row["requires_grad"] != parameters[row["name"]].requires_grad for row in rows):
        raise ValueError("Trainable/frozen configuration changed after inventory")
    eligible = {row["name"] for row in rows if row["regularized"]}
    values = {key: {} for key in ("combined", "data_inferred", "l1", "l2")}
    unobserved = []
    for name, parameter in parameters.items():
        grad = parameter.grad
        if grad is not None and grad.is_sparse:
            raise ValueError("Sparse gradients need a separate audited accounting policy")
        weight = parameter.detach().to(dtype=torch.float64)
        one = weight.sign() * l1 if name in eligible else torch.zeros_like(weight)
        two = weight * l2 if name in eligible else torch.zeros_like(weight)
        if grad is None:
            for component in values.values():
                component[name] = None
            if name in eligible:
                unobserved.append({"name": name, "expected_l1_gradient_l2": float(one.norm().item()),
                                   "expected_l2_gradient_l2": float(two.norm().item()),
                                   "expected_penalty_l2": float((one + two).norm().item()),
                                   "active_penalty_expected": bool(l1 or l2),
                                   "reason": "eligible_but_grad_None_penalty_participation_unconfirmed"})
            continue
        combined = grad.detach().to(dtype=torch.float64)
        values["combined"][name] = combined
        values["l1"][name], values["l2"][name] = one, two
        values["data_inferred"][name] = combined - one - two

    def summarize(component):
        triples = [(name, component[name], parameter.requires_grad)
                   for name, parameter in parameters.items()]
        grouped = {key: _gradient_values_summary([
            (name, component[name], parameters[name].requires_grad) for name in names])
            for key, names in groups.items()}
        overall = _gradient_values_summary(triples)
        combined_norm = (math.sqrt(sum(item["l2"] ** 2 for item in grouped.values()))
                         if overall["all_finite"] else None)
        return {"groups": grouped, "global": overall,
                "sqrt_group_squared_norms": combined_norm,
                "norm_identity_pass": bool(overall["all_finite"] and math.isclose(
                    overall["l2"], combined_norm, rel_tol=1e-12, abs_tol=1e-12))}

    return {"components": {name: summarize(value) for name, value in values.items()},
            "unobserved_regularized_parameters": unobserved,
            "regularizer_participation_pass": not any(row["active_penalty_expected"] for row in unobserved),
            "stage": "post_accumulation_post_unscale_pre_clip",
            "regularizer_applications_assumed": 1, "adamw_decay_included": False,
            "numeric_caveat": "Inferred subtraction is subject to FP32 rounding and cancellation; not exact",
            "optimizer_update_verified": False}


def snapshot_adamw_update(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    """Snapshot immediately BEFORE step; AdamW skips tensors with grad=None.

    expected_decay_delta is a component of the update, not a loss gradient.
    A GradScaler-skipped step must be identified when reporting this snapshot.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("This decay diagnostic is defined for AdamW only")
    names = {id(p): name for name, p in _parameters(model).items()}
    records = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = names.get(id(parameter))
            if name is None or name in records:
                raise ValueError("Optimizer parameter unknown or duplicated")
            lr, wd = float(group["lr"]), float(group["weight_decay"])
            records[name] = {"before": parameter.detach().clone(), "lr": lr,
                             "weight_decay": wd, "had_gradient": parameter.grad is not None}
    return records


def parameter_update_statistics(model: nn.Module, snapshot: dict,
                                *, optimizer_step_succeeded: bool) -> dict:
    parameters = _parameters(model)
    rows = []
    before_ss = delta_ss = decay_ss = 0.0
    for name, record in snapshot.items():
        before = record["before"].to(dtype=torch.float64)
        after = parameters[name].detach().to(dtype=torch.float64)
        delta = after - before
        coefficient = (-record["lr"] * record["weight_decay"]
                       if record["had_gradient"] and optimizer_step_succeeded else 0.0)
        decay = coefficient * before
        old_ss = float(before.square().sum().item())
        update_ss = float(delta.square().sum().item())
        decay_part_ss = float(decay.square().sum().item())
        before_ss += old_ss
        delta_ss += update_ss
        decay_ss += decay_part_ss
        rows.append({"name": name, "actual_update_l2": math.sqrt(update_ss),
                     "expected_adamw_decay_delta_l2": math.sqrt(decay_part_ss),
                     "relative_actual_update": math.sqrt(update_ss / old_ss) if old_ss else None,
                     "had_gradient": record["had_gradient"], "lr": record["lr"],
                     "weight_decay": record["weight_decay"]})
    return {"parameters": rows, "actual_update_l2": math.sqrt(delta_ss),
            "expected_adamw_decay_delta_l2": math.sqrt(decay_ss),
            "relative_actual_update": math.sqrt(delta_ss / before_ss) if before_ss else None,
            "optimizer_step_succeeded": optimizer_step_succeeded}


def capture_rng() -> dict:
    return {"python": random.getstate(),
            "numpy": copy.deepcopy(np.random.get_state()) if np is not None else None,
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [x.clone() for x in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    if state["numpy"] is not None:
        if np is None:
            raise ValueError("Checkpoint requires NumPy RNG support")
        np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_initialized():
            raise ValueError("CUDA RNG restore requires already initialized matching devices")
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count changed")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@contextmanager
def preserve_eval(model: nn.Module):
    """Eval/no_grad with exact per-module mode and global RNG restoration.

    Does not cover caller-created torch.Generator objects or DataLoader/sampler
    cursors; those must be separately preserved in data_order_state.
    """
    modes = [(module, module.training) for module in model.modules()]
    rng = capture_rng()
    try:
        model.eval()
        with torch.no_grad():
            yield model
    finally:
        for module, flag in modes:
            module.training = flag
        restore_rng(rng)


def enable_selective_dropout(model: nn.Module, dropout_names: Sequence[str],
                             *, training: bool) -> dict:
    """After a classifier's train() override, enable ONLY whitelisted Dropout.

    All parent modules (including frozen MAE/Mamba) remain eval; this does not
    detach the graph or prevent prompt/last-stage parameter gradients.
    """
    modules = dict(model.named_modules())
    if len(dropout_names) != len(set(dropout_names)):
        raise ValueError("Duplicate Dropout whitelist path")
    for name in dropout_names:
        if name not in modules or type(modules[name]) is not nn.Dropout:
            raise ValueError(f"Explicit nn.Dropout module required: {name}")
    model.eval()
    for name in dropout_names:
        modules[name].train(training)
    return {name: {"class": type(module).__name__, "training": module.training}
            for name, module in modules.items()}


def weighted_microbatch_loss(mean_loss: torch.Tensor, microbatch_n: int,
                             actual_optimizer_batch_n: int) -> torch.Tensor:
    """For sum(mean_loss*n/N) over one step, including a partial final batch."""
    if not 0 < microbatch_n <= actual_optimizer_batch_n:
        raise ValueError("Positive actual sample counts required")
    return mean_loss * (microbatch_n / actual_optimizer_batch_n)


def state_tensor_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not torch.is_tensor(tensor):
            raise ValueError("Only tensor state is supported in this checkpoint schema")
        tensor = tensor.detach().cpu().contiguous()
        metadata = [name, str(tensor.dtype), list(tensor.shape)]
        digest.update(json.dumps(metadata, separators=(",", ":")).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _class_name(value):
    return (type(value).__module__ + "." + type(value).__qualname__
            if value is not None else None)


def make_step_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, *, mutable_state_names: Sequence[str],
    immutable_backbone_sha256: str, successful_steps: int, skipped_steps: int,
    epoch: int, next_batch_index: int, data_order_state: dict,
    scheduler=None, scaler=None, accumulation_index: int = 0,
) -> dict:
    """Construct a CPU payload at a clean optimizer-step boundary only.

    This is a schema/helper test, not proof the real NeuroSTORM runner resumes.
    A real runner must bind immutable_backbone_sha256 to its verified source file
    and explicitly restore sampler/generator/data-order state before continuing.
    """
    parameters = _parameters(model)
    if accumulation_index != 0 or any(p.grad is not None for p in parameters.values()):
        raise ValueError("Save only after a step and zero_grad(set_to_none=True)")
    if not re.fullmatch(r"[0-9a-f]{64}", immutable_backbone_sha256):
        raise ValueError("Verified immutable source SHA256 required")
    counters = (successful_steps, skipped_steps, epoch, next_batch_index)
    if any(type(value) is not int or value < 0 for value in counters):
        raise ValueError("Nonnegative integer step/epoch/cursor counters required")
    state = model.state_dict()
    mutable = set(mutable_state_names)
    if len(mutable) != len(mutable_state_names) or not mutable <= set(state):
        raise ValueError("Mutable state names must be unique existing state keys")
    if not {name for name, p in parameters.items() if p.requires_grad} <= mutable:
        raise ValueError("Every trainable parameter must be saved")
    known = {id(p): name for name, p in parameters.items()}
    optimizer_groups = [[known.get(id(p)) for p in group["params"]]
                        for group in optimizer.param_groups]
    optimizer_names = [name for group in optimizer_groups for name in group]
    if None in optimizer_names or len(optimizer_names) != len(set(optimizer_names)):
        raise ValueError("Optimizer must contain unique parameters from this model")
    if not set(optimizer_names) <= mutable:
        raise ValueError("Every optimizer-owned tensor must be saved as mutable state")
    frozen = {name: value for name, value in state.items() if name not in mutable}
    return {"schema": 1, "kind": "G0_STEP_BOUNDARY_HELPER_NOT_RUNNER_CERTIFICATE",
            "boundary": {"accumulation_index": 0, "gradient_buffers_cleared": True},
            "mutable_state": _cpu_copy({name: state[name] for name in sorted(mutable)}),
            "immutable_backbone_sha256": immutable_backbone_sha256,
            "immutable_state_sha256": state_tensor_hash(frozen),
            "optimizer_parameter_names": optimizer_names,
            "optimizer_parameter_groups": optimizer_groups,
            "trainable_parameter_names": [name for name, p in parameters.items() if p.requires_grad],
            "optimizer_class": _class_name(optimizer), "scheduler_class": _class_name(scheduler),
            "scaler_class": _class_name(scaler),
            "optimizer": _cpu_copy(optimizer.state_dict()),
            "scheduler": _cpu_copy(scheduler.state_dict()) if scheduler is not None else None,
            "scaler": _cpu_copy(scaler.state_dict()) if scaler is not None else None,
            "precision": "scaler_enabled" if scaler is not None else "no_GradScaler",
            "rng": capture_rng(), "data_order_state": _cpu_copy(data_order_state),
            "successful_optimizer_steps": successful_steps, "skipped_optimizer_steps": skipped_steps,
            "epoch": epoch, "next_batch_index": next_batch_index,
            "module_training": {name: module.training for name, module in model.named_modules()}}


def restore_step_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                            payload: dict, *, immutable_backbone_sha256: str,
                            scheduler=None, scaler=None) -> dict:
    """Validate immutable binding before loading mutable model/optimizer/RNG."""
    if payload.get("schema") != 1 or payload.get("boundary") != {
            "accumulation_index": 0, "gradient_buffers_cleared": True}:
        raise ValueError("Unsupported checkpoint schema or unsafe step boundary")
    if payload["immutable_backbone_sha256"] != immutable_backbone_sha256:
        raise ValueError("Immutable backbone source hash mismatch")
    state = model.state_dict()
    mutable = payload["mutable_state"]
    if not set(mutable) <= set(state):
        raise ValueError("Checkpoint keys do not belong to target model")
    frozen = {name: value for name, value in state.items() if name not in mutable}
    if state_tensor_hash(frozen) != payload["immutable_state_sha256"]:
        raise ValueError("Immutable model state mismatch")
    params = _parameters(model)
    trainable = [name for name, p in params.items() if p.requires_grad]
    if trainable != payload["trainable_parameter_names"]:
        raise ValueError("Target trainable/frozen configuration changed")
    if not set(trainable) <= set(mutable):
        raise ValueError("Target trainable state is absent from checkpoint")
    known = {id(p): name for name, p in params.items()}
    groups = [[known.get(id(p)) for p in group["params"]] for group in optimizer.param_groups]
    if groups != payload["optimizer_parameter_groups"]:
        raise ValueError("Optimizer parameter order or grouping changed")
    if any(payload[key + "_class"] != _class_name(value)
           for key, value in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler))):
        raise ValueError("Optimizer/scheduler/scaler implementation class changed")
    if (scheduler is None) != (payload["scheduler"] is None):
        raise ValueError("Scheduler presence changed")
    if (scaler is None) != (payload["scaler"] is None):
        raise ValueError("GradScaler presence changed")
    if set(payload["module_training"]) != set(dict(model.named_modules())):
        raise ValueError("Module structure changed")
    for name, value in mutable.items():
        if value.shape != state[name].shape or value.dtype != state[name].dtype:
            raise ValueError(f"Mutable shape/dtype changed: {name}")
    model.load_state_dict(mutable, strict=False)
    optimizer.load_state_dict(_cpu_copy(payload["optimizer"]))
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler"])
    optimizer.zero_grad(set_to_none=True)
    for name, module in model.named_modules():
        module.training = payload["module_training"][name]
    restore_rng(payload["rng"])
    return {key: copy.deepcopy(payload[key]) for key in (
        "successful_optimizer_steps", "skipped_optimizer_steps", "epoch", "next_batch_index",
        "data_order_state")}
