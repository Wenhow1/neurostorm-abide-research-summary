# 阅读副本：只展开压缩语句、调整空白并补充中文说明，函数与计算逻辑保持不变。
# 本文件不替换已经封定并正在运行的源码；原始文件、SHA256 与 AST 核验见同目录来源清单。
# 这不是单文件独立运行包：真实运行还需要原协议、旧受试者划分、数据、特征或检查点。
# 不要在阅读目录直接启动任务；部分路径与来源校验依赖原项目布局，__file__ 的位置也会改变。
# 任务：读取既有大小 Mamba / Transformer 检查点，冻结编码器，提取完整 FC 的分类特征。
# 输入：290 人完整 FC，每人为 116×116；标准化统计来自此前 610 人上游训练。
# 小模型输出：116 个脑区 × 288 通道；全局平均得到 288 维，按脑区再按通道展开得到 33,408 维。
# 大模型输出：116 个脑区 × 576 通道；全局平均得到 576 维，展开得到 66,816 维。
# 同时读取每个模型的预训练检查点与初始随机检查点，保持受试者、脑区与通道顺序一致。
# GAP 使用旧特征文件逐字节复制，并用当前冻结推理核对；flatten 保留全部已输出脑区特征。
# 下游路径只调用 encode_full：为严格加载检查点可以实例化完整模型，但重建解码器禁止调用。
# 本步骤无优化器、无 backward、不训练骨干或分类头；需要 FC、原始项目模块、检查点和 GPU1。
# 工程限制：GPU1 ≥80℃暂停，降至65℃后恢复；旧实验的其他参数、数据划分不改变。
# 原始文件：feature_extract.py
# 原始 SHA256：be4340465ef254b812f60a1ba3a7ee6c3718631df2f508473e137de4acff0461

"""Frozen complete-FC feature extraction. No optimizer, backward, or decoder call.

One job is an arm/seed pair and reads the exact trained and initialization files
from the previous full610 refit. GAP is copied byte-for-byte from the previous
experiment; complete ROI features are flattened in canonical ROI/channel order.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
GPU_UUID = 'GPU-7ac39891-108b-f897-b92f-cd51e5392bab'
PAUSE_C = 80.0
RESUME_C = 65.0
MEMORY_PAUSE_FRACTION = 0.9
WAIT_LIMIT_SECONDS = 120 * 60
GAP_ATOL = 2e-06
GAP_RTOL = 1e-05

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda : stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')

def resolve(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


# 读取指定 GPU1 的身份、温度与显存，不查询或改变其他用户任务。
def query_gpu():
    """Fail closed on unreadable temperature or device identity."""
    output = subprocess.check_output(['nvidia-smi', '-i', GPU_UUID, '--query-gpu=index,uuid,temperature.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
        text=True,
        timeout=15).strip()
    lines = output.splitlines()
    if len(lines) != 1:
        raise RuntimeError('Expected exactly one GPU telemetry row')
    fields = [x.strip() for x in lines[0].split(',')]
    if len(fields) != 5 or fields[1] != GPU_UUID or int(fields[0]) != 1:
        raise RuntimeError(f'GPU telemetry identity mismatch: {output}')
    (temperature, used, total) = map(float, fields[2:])
    if not all((math.isfinite(x) for x in (temperature, used, total))) or total <= 0 or used < 0:
        raise RuntimeError('Invalid GPU telemetry values')
    return {'physical_index': 1,
        'uuid': GPU_UUID,
        'temperature_C': temperature,
        'memory_used_MiB': used,
        'memory_total_MiB': total,
        'memory_fraction': used / total,
        'time_unix': time.time()}


# 温度与显存保护；只让本次特征推理等待，不重启模型训练或修改硬件设置。
class ThermalGuard:

    def __init__(self,
        status_path,
        events_path,
        context,
        query=query_gpu,
        sleeper=time.sleep,
        clock=time.monotonic,
        wait_limit=WAIT_LIMIT_SECONDS):
        self.status_path = Path(status_path)
        self.events_path = Path(events_path)
        self.context = dict(context)
        self.query = query
        self.sleeper = sleeper
        self.clock = clock
        self.wait_limit = wait_limit
        self.wait_seconds = 0.0
        self.samples = 0
        self.max_temperature = None
        self.max_memory_fraction = 0.0

    def check(self, **progress):
        telemetry = self.query()
        hot = telemetry['temperature_C'] >= PAUSE_C
        memory_full = telemetry['memory_fraction'] >= MEMORY_PAUSE_FRACTION
        waiting = hot or memory_full
        started = self.clock()
        if waiting:
            append(self.events_path,
                {**self.context, 'event': 'PAUSE', **progress, **telemetry, 'temperature_trigger': hot, 'memory_trigger': memory_full})
        while True:
            self.samples += 1
            t = telemetry['temperature_C']
            self.max_temperature = t if self.max_temperature is None else max(self.max_temperature, t)
            self.max_memory_fraction = max(self.max_memory_fraction, telemetry['memory_fraction'])
            hot = t > RESUME_C if hot else t >= PAUSE_C
            memory_full = telemetry['memory_fraction'] >= MEMORY_PAUSE_FRACTION
            if not hot and (not memory_full):
                break
            elapsed = self.clock() - started
            if self.wait_seconds + elapsed >= self.wait_limit:
                raise TimeoutError('Thermal/memory waiting exceeded cumulative 120 minute budget')
            put(self.status_path,
                {**self.context, 'status': 'WAITING', **progress, 'reason': 'temperature' if hot else 'memory', 'temperature_pause_C': PAUSE_C, 'temperature_resume_C': RESUME_C, 'memory_pause_fraction': MEMORY_PAUSE_FRACTION, 'wait_seconds': self.wait_seconds + elapsed, **telemetry})
            self.sleeper(5)
            telemetry = self.query()
        if waiting:
            self.wait_seconds += self.clock() - started
            append(self.events_path,
                {**self.context, 'event': 'RESUME', **progress, 'wait_seconds': self.wait_seconds, **telemetry})
        put(self.status_path,
            {**self.context, 'status': 'RUNNING', **progress, 'wait_seconds': self.wait_seconds, **telemetry})
        return telemetry

def atomic_npy(path, values, np):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        np.save(stream, values, allow_pickle=False)
    tmp.replace(path)


# 核心读出：保留全部 116 个脑区，不平均，按固定脑区/通道顺序展开。
def flatten_tokens(encoded, d, np):
    if encoded.ndim != 4 or encoded.shape[1:] != (1, 116, d):
        raise AssertionError(f'Expected B x 1 x 116 x {d}, got {encoded.shape}')
    z = encoded[:, 0]
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite encoder features')
    return np.ascontiguousarray(z.reshape(len(z), 116 * d, order='C'))


# 核对 610 人上游归一化来源与 290 人下游名单无重叠，不重新划分。
def validate_source(raw, ids, norm, np):
    if raw.shape != (290, 116, 116) or len(ids) != 290 or ids.ndim != 1:
        raise AssertionError('Expected 290 aligned complete FCs and one-dimensional IDs')
    if len(set(map(str, ids))) != 290:
        raise AssertionError('Duplicate downstream ID')
    fit_ids = list(map(str, norm['fit_subject_ids']))
    if len(fit_ids) != 610 or len(set(fit_ids)) != 610:
        raise AssertionError('Input normalization must be fitted on exactly 610 unique upstream subjects')
    if set(fit_ids) & set(map(str, ids)):
        raise AssertionError('Upstream/downstream overlap')
    if not np.isfinite(raw).all() or not math.isfinite(float(norm['mean'])) or (not math.isfinite(float(norm['std']))) or (float(norm['std']) <= 0):
        raise AssertionError('Invalid FC data or upstream normalization')


# 仅恢复来源、协议和特征文件校验值全部一致的完整提取记录。
def check_resume(receipt_path, signature, root):
    receipt_path = Path(receipt_path)
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('status') != 'COMPLETE' or receipt.get('signature') != signature:
        raise RuntimeError('Existing completion receipt does not match this extraction contract')
    for record in receipt['records']:
        path = resolve(root, record['path'])
        if not path.is_file() or sha(path) != record['feature_sha256']:
            raise RuntimeError(f'Completed feature failed checksum: {path}')
    return receipt

def run(args):
    protocol_path = Path(args.protocol).resolve()
    root = protocol_path.parent
    protocol = json.loads(protocol_path.read_text())
    source = resolve(root, protocol['data']['source_root'])
    batch = int(protocol.get('feature_extraction', {}).get('batch_size', 4))
    if batch not in (2, 4):
        raise ValueError('Audited feature batch must be 2 or 4')
    name = f'{args.arm}_s{args.seed}'
    mode = 'pilot' if args.pilot else 'features'
    status = root / 'feature_status' / f'{name}_{mode}.json'
    events = root / 'feature_events' / f'{name}_{mode}.jsonl'
    lock_path = root / 'feature_status' / f'{name}_{mode}.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return run_locked(args,
                root,
                source,
                protocol_path,
                protocol,
                batch,
                name,
                mode,
                status,
                events)
        except BaseException as exc:
            fail = {'status': 'FAILED',
                'name': name,
                'mode': mode,
                'error': repr(exc),
                'traceback': traceback.format_exc(),
                'time_unix': time.time()}
            put(status, fail)
            append(events, fail)
            raise


# 主要执行流程：检查来源和权重 → 冻结完整模型 → 仅调用编码器 → 检查平均读出一致性 → 保存展开特征。
def run_locked(args, root, source, protocol_path, protocol, batch, name, mode, status, events):
    import numpy as np
    refit = source / 'refits' / f'{args.arm}_full610_s{args.seed}'
    ckpts = {'pretrained': refit / 'last.pt', 'random': refit / 'initial.pt'}
    fc_path = source / 'downstream_data' / 'FC.npy'
    ids_path = source / 'downstream_data' / 'IDs.npy'
    old_receipt_path = source / 'features' / f'{name}.json'
    original = json.loads(old_receipt_path.read_text())
    if original.get('status') != 'COMPLETE':
        raise AssertionError('Original GAP extraction incomplete')
    completion = json.loads((refit / 'completion.json').read_text())
    if completion.get('status') != 'COMPLETE':
        raise AssertionError('Original refit incomplete')
    checkpoints = {kind: sha(path) for (kind, path) in ckpts.items()}
    if completion.get('last_sha256') != checkpoints['pretrained']:
        raise AssertionError('Refit completion checkpoint digest mismatch')
    ids_hash = sha(ids_path)
    if original['ids_sha256'] != ids_hash:
        raise AssertionError('Original GAP ID order hash mismatch')
    source_records = {}
    for kind in ckpts:
        key = f'{name}_{kind}'
        records = [r for r in original['records'] if r['name'] == key]
        if len(records) != 1:
            raise AssertionError('Missing/ambiguous original GAP record')
        record = records[0]
        path = source / 'features' / f'{key}.npy'
        if record['checkpoint_sha256'] != checkpoints[kind] or record['feature_sha256'] != sha(path):
            raise AssertionError(f'Original {kind} feature/checkpoint hashes disagree')
        source_records[kind] = {**record, 'gap_path': str(path)}
    source_code = {filename: sha(source / filename) for filename in ['worker.py',
        'models.py',
        'base_models.py',
        'roi_small.py',
        'roi_large.py',
        'features.py']}
    sigmat = {'protocol_sha256': sha(protocol_path),
        'extraction_code_sha256': sha(__file__),
        'source_code_sha256': source_code,
        'checkpoints': checkpoints,
        'fc_sha256': sha(fc_path),
        'ids_sha256': ids_hash,
        'original_receipt_sha256': sha(old_receipt_path),
        'arm': args.arm,
        'seed': args.seed,
        'batch_size': batch,
        'pilot': bool(args.pilot),
        'gpu_uuid': GPU_UUID,
        'atol': GAP_ATOL,
        'rtol': GAP_RTOL}
    signature = hashlib.sha256(json.dumps(sigmat, sort_keys=True).encode()).hexdigest()
    receipt_path = root / ('pilot' if args.pilot else 'receipts') / f'{name}.json'
    if not args.pilot:
        previous = check_resume(receipt_path, signature, root)
        if previous:
            put(status, {**previous, 'resume_verified': True})
            return previous
    config_path = root / 'feature_status' / f'{name}_{mode}_config.json'
    if config_path.exists() and json.loads(config_path.read_text())['signature'] != signature:
        raise RuntimeError('Existing partial extraction uses another protocol/source; choose a new output directory')
    put(config_path, {'signature': signature, 'provenance': sigmat})
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        raise RuntimeError('Supervisor must set CUDA_VISIBLE_DEVICES to the physical GPU1 UUID only')
    context = {'name': name, 'arm': args.arm, 'seed': args.seed, 'mode': mode}
    guard = ThermalGuard(status, events, context)
    guard.check(phase='before_model_load')
    sys.path.insert(0, str(source))
    worker = importlib.import_module('worker')
    models = importlib.import_module('models')
    if Path(worker.__file__).resolve() != source / 'worker.py' or Path(models.__file__).resolve() != source / 'models.py':
        raise AssertionError('Imported worker/models from unexpected directory')

    def stop(signum, frame):
        raise InterruptedError(f'Feature extraction interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGUSR1, stop)
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible CUDA GPU')
    ck = torch.load(ckpts['pretrained'], map_location='cpu', weights_only=False)
    norm = ck['norm']
    del ck
    raw = np.load(fc_path, allow_pickle=False)
    ids = np.load(ids_path, allow_pickle=False)
    validate_source(raw, ids, norm, np)
    limit = 8 if args.pilot else 290
    d = 576 if args.arm.endswith('L') else 288
    model = models.make(5, args.seed, args.arm).cuda().eval()
    model.requires_grad_(False)
    decoder_calls = [0]

    def unexpected_decoder(module, inputs):
        decoder_calls[0] += 1
        raise AssertionError('Decoder must never be invoked during frozen feature extraction')
    hook = model.decoder.register_forward_pre_hook(unexpected_decoder)
    start = time.monotonic()
    records = []
    checks = []
    try:
        for (kind, checkpoint_path) in ckpts.items():
            guard.check(phase='checkpoint_load', kind=kind)
            if sha(checkpoint_path) != checkpoints[kind]:
                raise AssertionError('Checkpoint changed before inference')
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'], strict=True)
            del checkpoint
            model.eval()
            model.requires_grad_(False)
            before = worker.old.state_hash(model)
            pieces = []
            gap_pieces = []
            original_gap = np.load(source_records[kind]['gap_path'], allow_pickle=False)
            if original_gap.shape != (290, d) or not np.isfinite(original_gap).all():
                raise AssertionError('Unexpected original GAP matrix')
            for offset in range(0, limit, batch):
                guard.check(phase='inference',
                    kind=kind,
                    completed_subjects=offset,
                    total_subjects=limit)
                end = min(offset + batch, limit)
                with torch.inference_mode():
                    inputs = torch.from_numpy(((raw[offset:end] - norm['mean']) / norm['std']).astype('float32'))[:,
                        None].cuda()
                    encoded = model.encode_full(inputs)
                    if tuple(encoded.shape) != (end - offset, 1, 116, d):
                        raise AssertionError(f'Unexpected encoder tensor shape: {tuple(encoded.shape)}')
                    gap_pieces.append(encoded.mean((1, 2)).cpu().numpy())
                    pieces.append(encoded[:, 0].cpu().numpy())
                    del inputs, encoded
                torch.cuda.synchronize()
            roi = np.concatenate(pieces)
            gap = np.concatenate(gap_pieces)
            flat = flatten_tokens(roi[:, None], d, np)
            difference = float(np.abs(gap - original_gap[:limit]).max())
            if not np.allclose(gap, original_gap[:limit], atol=GAP_ATOL, rtol=GAP_RTOL):
                raise AssertionError(f'GAP parity failed: max absolute difference={difference}; tolerance unchanged')
            if not np.array_equal(flat.reshape(limit, 116, d), roi):
                raise AssertionError('Flatten ROI order changed')
            after = worker.old.state_hash(model)
            if before != after or sha(checkpoint_path) != checkpoints[kind] or decoder_calls[0] != 0:
                raise AssertionError('Frozen model/checkpoint or decoder-call invariant failed')
            if any((p.requires_grad or p.grad is not None for p in model.parameters())):
                raise AssertionError('Model contains trainable parameters or accumulated gradients')
            key = f'{name}_{kind}'
            check = {'kind': kind,
                'checkpoint_path': str(checkpoint_path),
                'checkpoint_sha256': checkpoints[kind],
                'model_state_before': before,
                'model_state_after': after,
                'source_gap_sha256': source_records[kind]['feature_sha256'],
                'gap_max_absolute_difference': difference,
                'gap_allclose_atol': GAP_ATOL,
                'gap_allclose_rtol': GAP_RTOL,
                'shape': [limit, 1, 116, d],
                'flatten_shape': list(flat.shape),
                'frozen_verified': True,
                'decoder_calls': decoder_calls[0]}
            checks.append(check)
            if not args.pilot:
                features = root / protocol.get('feature_dir', 'features')
                features.mkdir(parents=True, exist_ok=True)
                gap_path = features / f'{key}_gap.npy'
                flat_path = features / f'{key}_flatten.npy'
                temporary = gap_path.with_name(gap_path.name + '.tmp')
                shutil.copyfile(source_records[kind]['gap_path'], temporary)
                if sha(temporary) != source_records[kind]['feature_sha256']:
                    raise AssertionError('Copied GAP bytes differ')
                temporary.replace(gap_path)
                atomic_npy(flat_path, flat, np)
                for (readout,
                    path,
                    shape) in [('gap', gap_path, [290, d]),
                    ('flatten', flat_path, [290, 116 * d])]:
                    records.append({'name': f'{key}_{readout}',
                        'readout': readout,
                        'kind': kind,
                        'path': str(path.relative_to(root)),
                        'shape': shape,
                        'dtype': 'float32',
                        'feature_sha256': sha(path),
                        'checkpoint_sha256': checkpoints[kind],
                        'ids_sha256': ids_hash,
                        'gap_copied_exactly': readout == 'gap'})
            del roi, gap, flat, pieces, gap_pieces, original_gap
            torch.cuda.empty_cache()
    finally:
        hook.remove()
        del model
        torch.cuda.empty_cache()
    if sha(fc_path) != sigmat['fc_sha256'] or sha(ids_path) != ids_hash:
        raise AssertionError('Source FC/ID data changed during inference')
    if any((sha(path) != checkpoints[kind] for (kind, path) in ckpts.items())):
        raise AssertionError('Checkpoint changed during extraction')
    receipt = {'status': 'PILOT_PASS' if args.pilot else 'COMPLETE',
        **context,
        'signature': signature,
        'provenance': sigmat,
        'subjects': limit,
        'records': records,
        'checks': checks,
        'subject_ids_in_output_order': [str(x) for x in ids[:limit]],
        'ids_sha256': ids_hash,
        'ordering': 'original downstream IDs; canonical ROI index then channel; NumPy C-order',
        'normalization': {'mean': norm['mean'], 'std': norm['std'], 'fit_subject_count': 610, 'fit_subject_ids_sha256': hashlib.sha256(json.dumps(norm['fit_subject_ids']).encode()).hexdigest()},
        'no_training': True,
        'backward_called': False,
        'decoder_calls': decoder_calls[0],
        'labels_read': False,
        'upstream_holdout_accessed': False,
        'batch_size': batch,
        'gpu_uuid': GPU_UUID,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'thermal_wait_seconds': guard.wait_seconds,
        'max_temperature_C': guard.max_temperature,
        'max_memory_fraction': guard.max_memory_fraction,
        'elapsed_seconds': time.monotonic() - start,
        'time_unix': time.time()}
    put(receipt_path, receipt)
    put(status, receipt)
    append(events,
        {**context, 'event': receipt['status'], 'time_unix': time.time(), 'elapsed_seconds': receipt['elapsed_seconds'], 'thermal_wait_seconds': guard.wait_seconds})
    return receipt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', default=str(Path(__file__).with_name('protocol.json')))
    parser.add_argument('--arm', required=True, choices=['MS', 'TS', 'ML', 'TL'])
    parser.add_argument('--seed', required=True, type=int, choices=[1, 2])
    parser.add_argument('--pilot',
        action='store_true',
        help='Only first eight subjects; validate both checkpoint kinds, write no feature arrays')
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({'status': result['status'], 'name': result['name'], 'mode': result['mode'], 'subjects': result['subjects']}, ensure_ascii=False),
        flush=True)
if __name__ == '__main__':
    main()
