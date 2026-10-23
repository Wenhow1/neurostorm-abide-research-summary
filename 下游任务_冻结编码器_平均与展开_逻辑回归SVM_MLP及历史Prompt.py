"""
下游输入是真实完整FC，经冻结编码器提取特征；不调用重建解码器。
小模型116×288：平均288维，展开33408维；大模型116×576：平均576维，展开66816维。
逻辑回归：无线性层后的隐藏激活，概率由sigmoid关系得到；线性SVM：无隐藏激活，不输出校准概率。
小MLP：单隐藏层32单元ReLU，二分类概率输出。完整类参数与内层选参见D2。
训练折内标准化，10外层×3内层；C六档0.001/0.003/0.01/0.03/0.1/1。
D3/D4是历史小块Prompt设计，允许训练提示参数和分类器，与当前冻结展开对照不同。
阅读顺序D1特征→D2分类→按需看D3/D4。对应报告08至11。

用途：模型与任务设计阅读，不是独立运行入口。
本文件按章节汇集真实来源模块的完整类/函数；没有执行模型、训练或读取数据。
各章节仍保留各自的import名称与类名，同名类属于不同版本，不能顺序装配。
Mamba底层选择性状态空间算子来自mamba_ssm库；PyTorch/scikit-learn基础算子不在此重复实现。
完整可恢复工程、源文件位置和校验证据见私有完整归档。
"""

raise RuntimeError("这是代码阅读文件，请勿直接执行；运行使用原工程。")


# ========================================================================
# D1 完整FC冻结特征提取与平均 展开
# 来源：冻结脑区Mamba_Transformer_平均288或576_展开33408或66816.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

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

def flatten_tokens(encoded, d, np):
    if encoded.ndim != 4 or encoded.shape[1:] != (1, 116, d):
        raise AssertionError(f'Expected B x 1 x 116 x {d}, got {encoded.shape}')
    z = encoded[:, 0]
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite encoder features')
    return np.ascontiguousarray(z.reshape(len(z), 116 * d, order='C'))

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


# ========================================================================
# D2 三种分类器 标准化 内层选择与外层评估
# 来源：逻辑回归_线性SVM_单隐藏层32ReLU_MLP_十折分类.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

import os

import argparse

import csv

import fcntl

import hashlib

import json

from pathlib import Path

import re

import sys

import time

import traceback

import warnings

import joblib

import numpy as np

import sklearn

from sklearn.exceptions import ConvergenceWarning

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss, roc_auc_score

from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC

from threadpoolctl import threadpool_limits

C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 1.0]

ALPHA_GRID = [0.001, 0.01, 0.1]

HEAD_ALIASES = {'lr': 'lr',
    'logistic': 'lr',
    'logistic_regression': 'lr',
    'svm': 'svm',
    'linear_svm': 'svm',
    'linear_svc': 'svm',
    'mlp': 'mlp'}

def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for (k, v) in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and (not np.isfinite(value)):
        raise ValueError('Nonfinite JSON value')
    return value

def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda : stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def csvout(path, rows):
    if not rows:
        raise ValueError('Cannot write empty result table')
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)

def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch('[A-Za-z0-9_.-]+', value) or value in ('.', '..'):
        raise ValueError(f'Unsafe task/feature name: {value!r}')
    return value

def normalize_head(head):
    try:
        return HEAD_ALIASES[head]
    except KeyError:
        raise ValueError(f'Unsupported head: {head}; choose lr/svm/mlp')

def resolved(root, value):
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()

def classifier_spec(head, value, seed, sample_count, feature_count):
    head = normalize_head(head)
    if head == 'lr':
        return LogisticRegression(C=float(value),
            solver='liblinear',
            penalty='l2',
            max_iter=20000,
            tol=1e-06,
            random_state=seed)
    if head == 'svm':
        version = tuple((int(v) for v in re.match('(\\d+)\\.(\\d+)', sklearn.__version__).groups()))
        dual = 'auto' if version >= (1, 3) else bool(sample_count <= feature_count)
        return LinearSVC(C=float(value),
            penalty='l2',
            loss='squared_hinge',
            max_iter=20000,
            dual=dual,
            random_state=seed)
    return MLPClassifier(hidden_layer_sizes=(32,),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        alpha=float(value),
        batch_size=32,
        max_iter=200,
        early_stopping=False,
        n_iter_no_change=20,
        tol=0.0001,
        random_state=seed)

def parameter_grid(head):
    return list(ALPHA_GRID if normalize_head(head) == 'mlp' else C_GRID)

def parameter_name(head):
    return 'alpha' if normalize_head(head) == 'mlp' else 'C'

def choose_candidate(choices, head):
    best = max((row['mean_inner_AUROC'] for row in choices))
    tied = [row for row in choices if best - row['mean_inner_AUROC'] <= 1e-12]
    return (max if normalize_head(head) == 'mlp' else min)(tied, key=lambda row: row['value'])

def score_model(model, x, head):
    if normalize_head(head) == 'svm':
        scores = np.asarray(model.decision_function(x), dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite SVM decision score')
        return (scores, (scores >= 0).astype(int), None)
    probabilities = np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError('Invalid predicted probabilities')
    return (probabilities, (probabilities >= 0.5).astype(int), probabilities)

def metrics(y, scores, predicted, probabilities):
    (tn, fp, fn, tp) = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {'AUROC': float(roc_auc_score(y, scores)),
        'ACC': float(accuracy_score(y, predicted)),
        'BA': float(balanced_accuracy_score(y, predicted)),
        'BCE': None if probabilities is None else float(log_loss(y, probabilities, labels=[0, 1])),
        'Brier': None if probabilities is None else float(np.mean((probabilities - y) ** 2)),
        'sensitivity': float(tp / (tp + fn)),
        'specificity': float(tn / (tn + fp)),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp)}

def index_set(values, n, label):
    arr = np.asarray(values)
    if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f'{label}: integer indices required')
    if not len(arr) or np.min(arr) < 0 or np.max(arr) >= n:
        raise ValueError(f'{label}: invalid indices')
    if len(np.unique(arr)) != len(arr):
        raise ValueError(f'{label}: duplicate indices')
    return set(arr.tolist())

def validate_splits(splits, y, ids, outer=10, inner=3, expected_n=290):
    n = len(y)
    if n != expected_n or len(ids) != n:
        raise ValueError(f'Expected {expected_n} aligned subjects, found {n}')
    if np.asarray(y).ndim != 1 or set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError('Binary labels 0/1 required')
    if np.asarray(ids).ndim != 1 or len(set((str(v) for v in ids))) != n:
        raise ValueError('Subject IDs must be unique and one dimensional')
    if len(splits) != outer:
        raise ValueError(f'Expected {outer} outer folds')
    all_indices = set(range(n))
    test_counts = np.zeros(n, dtype=int)
    fold_names = set()
    for sp in splits:
        f = sp['fold']
        if not isinstance(f, int) or f < 0 or f in fold_names:
            raise ValueError('Fold identifiers must be unique nonnegative integers')
        fold_names.add(f)
        tr = index_set(sp['train'], n, f'fold{f}/train')
        te = index_set(sp['test'], n, f'fold{f}/test')
        if tr & te or tr | te != all_indices:
            raise ValueError(f'fold{f}: outer partition leakage/gap')
        if expected_n == 290 and (len(tr) != 261 or len(te) != 29):
            raise ValueError('Production split must be 261 train / 29 test')
        test_counts[list(te)] += 1
        if len(sp['inner']) != inner:
            raise ValueError(f'fold{f}: expected {inner} inner folds')
        valid_counts = {i: 0 for i in tr}
        for (k, ii) in enumerate(sp['inner']):
            it = index_set(ii['train'], n, f'fold{f}/inner{k}/train')
            iv = index_set(ii['validation'], n, f'fold{f}/inner{k}/validation')
            if it & iv or it | iv != tr or (it | iv) & te:
                raise ValueError(f'fold{f}/inner{k}: train/validation/test leakage or partition gap')
            for i in iv:
                valid_counts[i] += 1
            for (group, key) in ((it, 'inner_train'), (iv, 'inner_validation')):
                if len(np.unique(y[list(group)])) != 2:
                    raise ValueError(f'{f}/{k}/{key}: both classes required')
        if any((v != 1 for v in valid_counts.values())):
            raise ValueError(f'fold{f}: inner validation is not a once-only partition')
        if len(np.unique(y[list(te)])) != 2:
            raise ValueError(f'{f}: outer test requires both classes')
    if not np.all(test_counts == 1):
        raise ValueError('Every subject must be outer-tested exactly once')
    return {'subjects': n,
        'outer_folds': outer,
        'inner_folds': inner,
        'outer_test_once_only': True,
        'no_subject_overlap': True,
        'indices_are_global_not_inner_relative': True}

def fit_model(x, y, train, head, value, seed):
    train = np.asarray(train, dtype=int)
    classifier = classifier_spec(head, value, seed, len(train), x.shape[1])
    model = make_pipeline(StandardScaler(), classifier)
    start = time.time()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always')
        model.fit(x[train], y[train])
    recorded = [{'category': w.category.__name__, 'message': str(w.message)} for w in caught]
    learned_arrays = getattr(classifier, 'coefs_', [getattr(classifier, 'coef_', np.array([]))])
    if any((not np.isfinite(a).all() for a in learned_arrays)):
        raise ValueError('Nonfinite fitted classifier weights')
    if not np.allclose(model[0].mean_, x[train].mean(axis=0), rtol=1e-07, atol=1e-09):
        raise AssertionError('Scaler mean differs from fitting partition')
    if not np.all(np.isfinite(model[0].mean_)):
        raise ValueError('Nonfinite fitted scaler')
    diag = {'fit_seconds': time.time() - start,
        'seed': seed,
        'train_n': len(train),
        'scaler_fit_only_train': True,
        'scaler_n_samples_seen': clean_json(model[0].n_samples_seen_),
        'classifier_parameters': classifier.get_params(),
        'warnings': recorded,
        'convergence_warning': any((isinstance(w.message, ConvergenceWarning) for w in caught)),
        'n_iter': clean_json(getattr(classifier, 'n_iter_', None))}
    if normalize_head(head) == 'mlp':
        diag.update(loss_curve=clean_json(classifier.loss_curve_),
            final_training_loss=float(classifier.loss_),
            output_activation=classifier.out_activation_,
            no_validation_early_stopping=True,
            stop_rule='training loss tol=1e-4 / n_iter_no_change=20; max_iter=200',
            trainable_parameters=sum((a.size for a in classifier.coefs_)) + sum((a.size for a in classifier.intercepts_)))
        if classifier.out_activation_ != 'logistic' or len(classifier.coefs_) != 2:
            raise AssertionError('Expected one hidden ReLU layer and binary logistic output')
    return (model, diag)

def input_paths(root, protocol, job):
    data = protocol.get('data', {})
    source = resolved(root, data.get('source_root', '../cpac610_schedule_mechanism_20261023'))
    return {'features': resolved(root, protocol.get('feature_dir', 'features')) / (safe_name(job['feature_key']) + '.npy'),
        'labels': resolved(root, data.get('labels_path', str(source / 'downstream_data/Y.npy'))),
        'ids': resolved(root, data.get('ids_path', str(source / 'downstream_data/IDs.npy'))),
        'splits': resolved(root, data.get('splits_path', str(source / 'downstream_splits.json')))}

def build_context(protocol_path, job, test_mode=False):
    protocol_path = Path(protocol_path).resolve()
    root = protocol_path.parent
    protocol = json.loads(protocol_path.read_text())
    name = safe_name(job['name'])
    head = normalize_head(job['head'])
    paths = input_paths(root, protocol, job)
    for (key, p) in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f'Missing {key}: {p}')
    y = np.load(paths['labels'], allow_pickle=False)
    ids = np.load(paths['ids'], allow_pickle=False)
    x = np.asarray(np.load(paths['features'], allow_pickle=False), dtype=np.float64)
    splits = json.loads(paths['splits'].read_text())
    if x.ndim != 2 or x.shape[0] != len(y) or x.shape[1] == 0 or (not np.isfinite(x).all()):
        raise ValueError('Features must be a finite subjects x features array aligned with labels')
    verification = validate_splits(splits,
        y,
        ids,
        outer=protocol.get('testing', {}).get('outer', 10) if test_mode else 10,
        inner=protocol.get('testing', {}).get('inner', 3) if test_mode else 3,
        expected_n=protocol.get('testing', {}).get('subjects', 290) if test_mode else 290)
    if bool(protocol.get('synthetic_test_only', False)) != bool(test_mode):
        raise ValueError('Synthetic test mode must be explicitly paired with synthetic_test_only protocol')
    hashes = {key: sha(p) for (key, p) in paths.items()}
    for (key, value) in protocol.get('input_hashes', {}).items():
        if key in hashes and hashes[key] != value:
            raise ValueError(f'{key}: immutable input hash mismatch')
    expected_feature = job.get('feature_sha256')
    if expected_feature and expected_feature != hashes['features']:
        raise ValueError('Feature hash mismatch')
    signature_material = {'job': job,
        'protocol_sha256': sha(protocol_path),
        'inputs': hashes,
        'code_sha256': sha(__file__),
        'sklearn_version': sklearn.__version__,
        'numpy_version': np.__version__,
        'synthetic_test_only': bool(test_mode)}
    signature = hashlib.sha256(json.dumps(signature_material, sort_keys=True).encode()).hexdigest()
    return (root,
        protocol,
        paths,
        x,
        y.astype(int),
        ids,
        splits,
        name,
        head,
        verification,
        signature_material,
        signature)

def fold_result_valid(fold_dir, signature, sp, ids, y):
    completion = fold_dir / 'completion.json'
    if not completion.exists():
        return None
    receipt = json.loads(completion.read_text())
    if receipt.get('signature') != signature:
        raise ValueError(f'{fold_dir}: resume signature mismatch')
    for (filename, digest) in receipt['artifact_sha256'].items():
        path = fold_dir / filename
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f'{fold_dir}: damaged completed artifact {filename}')
    result = json.loads((fold_dir / 'result.json').read_text())
    expected = list(map(int, sp['test']))
    if [r['index'] for r in result['predictions']] != expected:
        raise ValueError('Resumed test indices mismatch')
    for row in result['predictions']:
        i = row['index']
        if row['subject_id'] != str(ids[i]) or row['label'] != int(y[i]):
            raise ValueError('Resumed subject/label mismatch')
    return result

def run(protocol_path, job, test_mode=False, preflight=False):
    (root,
        protocol,
        paths,
        x,
        y,
        ids,
        splits,
        name,
        head,
        verification,
        sigmat,
        signature) = build_context(protocol_path,
        job,
        test_mode)
    if preflight:
        return {'status': 'PREFLIGHT_OK',
            'name': name,
            'head': head,
            'shape': list(x.shape),
            'split_validation': verification,
            'input_sha256': sigmat['inputs'],
            'fits_planned': len(splits) * (len(splits[0]['inner']) * len(parameter_grid(head)) + 1),
            'MLP_input_layer_parameters': (x.shape[1] + 1) * 32 if head == 'mlp' else None}
    out = resolved(root, protocol.get('output_dir', 'downstream')) / name
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(out,
            root,
            protocol,
            paths,
            x,
            y,
            ids,
            splits,
            name,
            head,
            verification,
            sigmat,
            signature,
            job)

def run_locked(out,
    root,
    protocol,
    paths,
    x,
    y,
    ids,
    splits,
    name,
    head,
    verification,
    sigmat,
    signature,
    job):
    previous = out / 'config.json'
    if previous.exists() and json.loads(previous.read_text()).get('signature') != signature:
        raise ValueError('Existing output belongs to another input/protocol/code; choose a new task name')
    put(previous,
        {'name': name, 'head': head, 'job': job, 'feature_shape': list(x.shape), 'signature': signature, 'provenance': sigmat, 'input_paths': {k: str(v) for (k, v) in paths.items()}, 'split_validation': verification, 'parameter_grid': parameter_grid(head), 'selection': 'highest mean inner AUROC; ties within 1e-12 choose stronger regularization', 'probability_available': head != 'svm', 'BCE_note': 'LinearSVC has decision scores, no probabilities or BCE' if head == 'svm' else 'BCE from predict_proba', 'threshold': 0 if head == 'svm' else 0.5, 'parallelism': 'single BLAS/OpenMP thread per process'})
    results = []
    for sp in splits:
        f = int(sp['fold'])
        fold_dir = out / f'fold_{f:02d}'
        fold_dir.mkdir(exist_ok=True)
        resumed = fold_result_valid(fold_dir, signature, sp, ids, y)
        if resumed is not None:
            results.append(resumed)
            continue
        try:
            put(out / 'status.json',
                {'status': 'RUNNING', 'fold': f, 'completed_folds': len(results), 'time': time.time()})
            scores = []
            choices = []
            diagnostics = []
            for value in parameter_grid(head):
                values = []
                for (k, ii) in enumerate(sp['inner']):
                    put(out / 'status.json',
                        {'status': 'RUNNING', 'fold': f, 'inner': k, parameter_name(head): value, 'completed_folds': len(results), 'time': time.time()})
                    (model, diag) = fit_model(x, y, ii['train'], head, value, 2026102100 + f * 10 + k)
                    diag.update(phase='inner',
                        fold=f,
                        inner=k,
                        parameter=parameter_name(head),
                        value=value)
                    diagnostics.append(diag)
                    put(fold_dir / 'fit_diagnostics.json', diagnostics)
                    if diag['convergence_warning'] and head != 'mlp':
                        raise RuntimeError('Linear classifier did not converge; diagnostic saved, no silent score accepted')
                    iv = np.asarray(ii['validation'], dtype=int)
                    (score, pred, prob) = score_model(model, x[iv], head)
                    val = metrics(y[iv], score, pred, prob)
                    values.append(val['AUROC'])
                    scores.append({'fold': f,
                        'inner': k,
                        parameter_name(head): value,
                        'train_n': len(ii['train']),
                        'validation_n': len(iv),
                        **val,
                        'convergence_warning': diag['convergence_warning']})
                    del model
                choices.append({'parameter': parameter_name(head),
                    'value': value,
                    'inner_AUROC': values,
                    'mean_inner_AUROC': float(np.mean(values))})
            chosen = choose_candidate(choices, head)
            selection = {'fold': f, 'chosen': chosen, 'candidates': choices, 'outer_test_used': False}
            put(fold_dir / 'selection.json', selection)
            csvout(fold_dir / 'inner_scores.csv', scores)
            tr = np.asarray(sp['train'], dtype=int)
            te = np.asarray(sp['test'], dtype=int)
            (model, diag) = fit_model(x, y, tr, head, chosen['value'], 2026102190 + f)
            diag.update(phase='outer_refit',
                fold=f,
                parameter=parameter_name(head),
                value=chosen['value'])
            diagnostics.append(diag)
            put(fold_dir / 'fit_diagnostics.json', diagnostics)
            if diag['convergence_warning'] and head != 'mlp':
                raise RuntimeError('Outer linear refit did not converge')
            (score, pred, prob) = score_model(model, x[te], head)
            (train_score, train_pred, train_prob) = score_model(model, x[tr], head)
            test_metrics = metrics(y[te], score, pred, prob)
            train_metrics = metrics(y[tr], train_score, train_pred, train_prob)
            predictions = [{'index': int(i),
                'subject_id': str(ids[i]),
                'fold': f,
                'label': int(y[i]),
                'score': float(score[j]),
                'predicted_label': int(pred[j]),
                'probability': None if prob is None else float(prob[j])} for (j,
                i) in enumerate(te)]
            result = {'fold': f,
                'selected_parameter': parameter_name(head),
                'selected_value': chosen['value'],
                'inner_selected_AUROC': chosen['mean_inner_AUROC'],
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'convergence_warning_fits': sum((d['convergence_warning'] for d in diagnostics)),
                'predictions': predictions}
            put(fold_dir / 'result.json', result)
            csvout(fold_dir / 'test_predictions.csv', predictions)
            tmp_model = fold_dir / 'classifier.joblib.tmp'
            joblib.dump(model, tmp_model)
            tmp_model.replace(fold_dir / 'classifier.joblib')
            del model
            filenames = ['selection.json',
                'inner_scores.csv',
                'fit_diagnostics.json',
                'result.json',
                'test_predictions.csv',
                'classifier.joblib']
            put(fold_dir / 'completion.json',
                {'status': 'COMPLETE', 'signature': signature, 'artifact_sha256': {n: sha(fold_dir / n) for n in filenames}, 'time': time.time()})
            results.append(result)
        except BaseException as error:
            failure = {'status': 'FAILED',
                'fold': f,
                'error': repr(error),
                'traceback': traceback.format_exc(),
                'time': time.time()}
            put(fold_dir / 'failure.json', failure)
            put(out / 'status.json', failure)
            raise
    predictions = sorted([p for result in results for p in result['predictions']],
        key=lambda p: p['index'])
    if [p['index'] for p in predictions] != list(range(len(y))):
        raise AssertionError('Incomplete/duplicate outer predictions')
    score = np.array([p['score'] for p in predictions])
    predicted = np.array([p['predicted_label'] for p in predictions])
    probability = None if head == 'svm' else np.array([p['probability'] for p in predictions])
    fold_rows = [{'fold': r['fold'],
        r['selected_parameter']: r['selected_value'],
        'inner_selected_AUROC': r['inner_selected_AUROC'],
        **r['test_metrics'],
        'train_AUROC': r['train_metrics']['AUROC'],
        'train_ACC': r['train_metrics']['ACC'],
        'train_BCE': r['train_metrics']['BCE'],
        'convergence_warning_fits': r['convergence_warning_fits']} for r in results]
    csvout(out / 'fold_results.csv', fold_rows)
    csvout(out / 'oof_predictions.csv', predictions)
    all_inner = []
    for r in results:
        with (out / f"fold_{r['fold']:02d}" / 'inner_scores.csv').open() as stream:
            all_inner.extend(csv.DictReader(stream))
    csvout(out / 'inner_scores.csv', all_inner)
    measured = metrics(y, score, predicted, probability)
    means = {key: None if measured[key] is None else float(np.mean([r['test_metrics'][key] for r in results])) for key in ['AUROC',
        'ACC',
        'BA',
        'BCE',
        'Brier',
        'sensitivity',
        'specificity']}
    completion = {'status': 'COMPLETE',
        'name': name,
        'head': head,
        'signature': signature,
        'subjects': len(y),
        'nested_folds': f"{len(splits)}x{len(splits[0]['inner'])}",
        'fits': len(splits) * (len(parameter_grid(head)) * len(splits[0]['inner']) + 1),
        'pooled': measured,
        'fold_equal_mean': means,
        'convergence_warning_fits': sum((r['convergence_warning_fits'] for r in results)),
        'probability_available': head != 'svm',
        'pooled_AUROC_caveat': 'OOF scores come from separately fitted fold models; inspect fold AUROC too, especially uncalibrated SVM margins',
        'time': time.time()}
    put(out / 'completion.json', completion)
    put(out / 'status.json', completion)
    return completion

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, help='JSON file containing name, feature_key, head')
    parser.add_argument('--protocol', default=str(Path(__file__).with_name('protocol.json')))
    parser.add_argument('--preflight',
        action='store_true',
        help='Validate real input alignment/shapes/splits; no classifier fitting')
    parser.add_argument('--synthetic-test',
        action='store_true',
        help='Only with an explicit synthetic_test_only protocol')
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text())
    try:
        result = run(args.protocol, job, test_mode=args.synthetic_test, preflight=args.preflight)
        print(json.dumps(clean_json(result), ensure_ascii=False, allow_nan=False), flush=True)
    except BaseException:
        traceback.print_exc()
        raise


# ========================================================================
# D3 历史小块模型与每块50个prompt
# 来源：15_旧小块Mamba_E1_D1完整结构与每块50个prompt.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

import sys, math

from pathlib import Path

import torch

from torch import nn

from torch.nn import functional as F

from neurostorm_2d.models.overlap_mae import OverlapMaskedAutoencoder2D, overlap_masked_mse

from neurostorm_2d.models.overlap_backbone import OverlapPatchMerging2D, OverlapPatchExpanding2D

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
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.register_parameter('prompt', None)

    def add_task_prompt(self, length):
        assert self.prompt is None
        self.prompt = nn.Parameter(self.norm1.weight.new_empty(1, length, self.dim))
        nn.init.trunc_normal_(self.prompt, std=0.02)

    def forward(self, x, *, use_prompts=False):
        b, h, w, c = x.shape
        wh, ww = (min(h, self.window[0]), min(w, self.window[1]))
        sh = self.shift[0] if h > self.window[0] else 0
        sw = self.shift[1] if w > self.window[1] else 0
        z = self.norm1(x)
        ph, pw = (-h % wh, -w % ww)
        z = F.pad(z, (0, 0, 0, pw, 0, ph))
        hp, wp = (h + ph, w + pw)
        if sh or sw:
            z = torch.roll(z, (-sh, -sw), (1, 2))
        z = z.reshape(b, hp // wh, wh, wp // ww, ww, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, wh * ww, c)
        if use_prompts:
            assert self.prompt is not None
            n = self.prompt.shape[1]
            z = torch.cat([self.prompt.expand(z.shape[0], -1, -1), z], dim=1)
            z = self.mamba(z)[:, n:]
        else:
            z = self.mamba(z)
        z = z.reshape(b, hp // wh, wp // ww, wh, ww, c).permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, c)
        if sh or sw:
            z = torch.roll(z, (sh, sw), (1, 2))
        z = x + z[:, :h, :w]
        return z + self.mlp(self.norm2(z))

class OfficialStage2D(nn.Module):

    def __init__(self, dim, shape, depth, window, downsample, full=False):
        super().__init__()
        self.blocks = nn.ModuleList([OfficialWindowBlock2D(dim,
             shape if full else window,
             0 if full or i % 2 == 0 else window // 2) for i in range(depth)])
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
            self.blocks.append(nn.ModuleList([OfficialWindowBlock2D(ds[i],
                 4,
                 0 if j % 2 == 0 else 2) for j in range(depths[i])]))
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
        self.encoder_kind, self.decoder_kind, self.mask_kind = (encoder, decoder, mask)
        dim = 16 if encoder == 'E0' else 36
        depths = (2, 2, 1, 1) if encoder == 'E0' else (2, 2, 6, 2)
        super().__init__(patch_size=patch,
             stride=stride,
             embed_dim=dim,
             depths=depths,
             window_sizes=(8,
             8,
             8,
             8),
             mask_unit=4,
             mask_policy='paired_symmetric')
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
        return self.head(self.mae.encode_full(image, use_prompts=self.mode != 'T0').mean((1, 2))).squeeze(-1)


# ========================================================================
# D4 历史Prompt参数冻结与装配
# 来源：14_旧小块Mamba_E1_D1_M1_R1R2_冻结与提示参数适配器.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

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

from instrumentation import capture_rng, gradient_statistics, regularization_inventory, restore_rng, state_tensor_hash

FIXED_MODEL = {'encoder': 'E1',
     'decoder': 'D1',
     'mask': 'M1',
     'patch': 4,
     'stride': 2,
     'ratio': 0.5,
     'mask_policy': 'paired_symmetric'}

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
    source_root = Path(__file__).resolve().parent.parent / 'model_source'
    path = Path(config.get('mechanisms_path', source_root / 'code_v1/mechanisms.py')).resolve()
    base = Path(config.get('base_code_path', path.parent.parent / 'base_code')).resolve()
    if not path.is_file() or not (base / 'neurostorm_2d/models/overlap_mae.py').is_file():
        raise FileNotFoundError('Byte-identical frozen mechanisms.py and base_code are required')
    expected = config.get('mechanisms_sha256')
    digest = _sha(path)
    if expected is not None and digest != expected:
        raise ValueError('Frozen mechanisms SHA256 mismatch')
    existing = sys.modules.get('neurostorm_2d')
    if existing is not None and existing.__file__ is not None:
        if not Path(existing.__file__).resolve().is_relative_to(base):
            raise ValueError('Another neurostorm_2d source tree is already imported in this process')
    sys.path.insert(0, str(base))
    name = '_currentfc_frozen_mechanisms_' + digest[:16]
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
    actual = Path(sys.modules['neurostorm_2d.models.overlap_mae'].__file__).resolve()
    if not actual.is_relative_to(base):
        raise ValueError('Imported base_code provenance differs from requested frozen source')
    return (module, {'mechanisms_path': str(path), 'mechanisms_sha256': digest, 'base_code_path': str(base)})

@contextmanager
def _mixer_context(module, factory):
    if factory is None:
        yield
        return
    with ExitStack() as stack:
        for target in (module,
             sys.modules['neurostorm_2d.models.mamba_block'],
             sys.modules['neurostorm_2d.models.overlap_backbone']):
            stack.enter_context(patch.object(target, '_real_mamba_factory', lambda: factory))
        yield

def _insert_mlp_dropout(block, prefix, probability):
    original = list(block.mlp.named_children())
    if [name for name,
         _ in original] != ['0',
         '1',
         '2'] or not isinstance(original[0][1],
         nn.Linear) or (not isinstance(original[1][1],
         nn.GELU)) or (not isinstance(original[2][1],
         nn.Linear)):
        raise ValueError(f'Unexpected frozen MLP layout: {prefix}')
    block.mlp = nn.Sequential(OrderedDict([('0',
         original[0][1]),
         ('1',
         original[1][1]),
         ('drop_after_gelu',
         nn.Dropout(probability)),
         ('2',
         original[2][1]),
         ('drop_after_linear',
         nn.Dropout(probability))]))
    return [prefix + '.mlp.drop_after_gelu', prefix + '.mlp.drop_after_linear']

class SelectiveClassifier(nn.Module):
    """Same MAE/head namespace and initialization as the frozen Classifier."""

    def __init__(self, original, dropout_paths, probability):
        super().__init__()
        self.mae, self.head, self.mode = (original.mae, original.head, original.mode)
        self.feature_dropout = nn.Dropout(probability) if probability else nn.Identity()
        self._dropout_paths = tuple(dropout_paths)
        if probability:
            self._dropout_paths += ('feature_dropout',)
        self.train(original.training)

    def train(self, mode=True):
        super().train(mode)
        self.mae.eval()
        modules = dict(self.named_modules())
        for name in self._dropout_paths:
            modules[name].train(mode)
        return self

    def forward_features(self, image):
        return self.mae.encode_full(image, use_prompts=self.mode in {'T1', 'T2'}).mean((1, 2))

    def forward(self, image):
        return self.head(self.feature_dropout(self.forward_features(image))).squeeze(-1)

def get_backbone(model):
    return model.mae if isinstance(model, SelectiveClassifier) else model

def _parameter_groups(model, mode):
    groups = {}
    for name, parameter in model.named_parameters():
        if mode != 'MAE':
            if name.split('.')[-1] == 'prompt':
                group = 'prompts'
            elif name.startswith('mae.encoder.stages.3.') and parameter.requires_grad:
                group = 'encoder_last_stage_nonprompt'
            elif name.startswith('head.'):
                group = 'classification_head'
            else:
                group = 'frozen_backbone'
        elif name.startswith('encoder.stages.'):
            group = 'encoder_stage_' + name.split('.')[2]
        elif name.startswith('encoder.pos_embeds.'):
            group = 'encoder_stage_' + name.split('.')[2]
        elif name.startswith('decoder.blocks.'):
            group = 'decoder_block_group_' + name.split('.')[2]
        elif name.startswith('decoder.first_patch_expanding.'):
            group = 'decoder_expansion_0'
        elif name.startswith('decoder.expands.'):
            group = 'decoder_expansion_' + str(int(name.split('.')[2]) + 1)
        elif name.startswith(('decoder.norm_up.', 'decoder.decoder_pred.')):
            group = 'decoder_prediction_head'
        elif name.startswith(('value_projection.', 'mask_projection.')) or name == 'mask_token':
            group = 'embedding'
        else:
            raise ValueError(f'Unregistered parameter namespace: {name}')
        groups.setdefault(group, []).append(name)
    if sum(map(len, groups.values())) != len(list(model.named_parameters())):
        raise ValueError('Nonexclusive parameter partition')
    return groups

def make_model(task_mode,
     arm,
     seed,
     config=None,
     *,
     pretrained_state=None,
     mechanisms_module=None,
     _test_mixer_factory=None):
    """Return (model, info); seed is the effective fold/seed initialization seed.

    MAE uses seed; classification uses seed for MAE construction and seed+800000
    for the original Prompt/head constructor, preserving the historical path.
    Optional config initialization_seed/classification_seed overrides are logged.
    pretrained_state must be the raw MAE state_dict, loaded strictly BEFORE
    prompts/head are created. The worker must hash/bind its checkpoint source.
    """
    config = dict(config or {})
    mode = str(task_mode).upper()
    if mode not in {'MAE', 'T0', 'T1', 'T2', 'T2_NO_PROMPT'}:
        raise ValueError('task_mode must be MAE/T0/T1/T2/T2_NO_PROMPT')
    arm_id = arm['id'] if isinstance(arm, dict) else str(arm)
    if arm_id not in {'A' + str(i) for i in range(8)}:
        raise ValueError('Only predeclared A0-A7 arms are accepted')
    for key, expected in FIXED_MODEL.items():
        if key in config and config[key] != expected:
            raise ValueError(f'Fixed model configuration changed: {key}')
    rate = 0.1 if arm_id in {'A6', 'A7'} and mode in {'MAE', 'T2'} else 0.0
    if isinstance(arm,
         dict) and 'dropout' in arm and (float(arm['dropout']) != (0.1 if arm_id in {'A6',
         'A7'} else 0.0)):
        raise ValueError('Arm Dropout probability differs from locked definition')
    if _test_mixer_factory is not None and (not config.get('allow_test_mixer', False)):
        raise ValueError('Synthetic mixer requires explicit allow_test_mixer=True')
    if mechanisms_module is None:
        module, source = load_frozen_mechanisms(config)
    else:
        if not config.get('allow_test_mixer', False):
            raise ValueError('Injected mechanisms module is test-only')
        module, source = (mechanisms_module, {'mechanisms_path': 'INJECTED_TEST_MODULE'})
    initial = int(config.get('initialization_seed', seed))
    classification_seed = int(config.get('classification_seed', seed + 800000))
    _seed_all(initial)
    with _mixer_context(module, _test_mixer_factory):
        mae = module.MechanismMAE(encoder='E1', decoder='D1', mask='M1', patch=4, stride=2)
    if mae.mask_ratio != 0.5 or mae.mask_policy != 'paired_symmetric':
        raise ValueError('Unexpected MAE masking contract')
    if pretrained_state is not None:
        mae.load_state_dict(pretrained_state, strict=True)
    dropout_paths = []
    if mode == 'MAE':
        model = mae
        if rate:
            for group, blocks in enumerate(mae.decoder.blocks):
                for block, layer in enumerate(blocks):
                    dropout_paths += _insert_mlp_dropout(layer, f'decoder.blocks.{group}.{block}', rate)
    else:
        _seed_all(classification_seed)
        original = module.Classifier(mae, 'T1' if mode == 'T0' else 'T2' if mode == 'T2_NO_PROMPT' else mode)
        if mode in {'T0', 'T2_NO_PROMPT'}:
            for block in mae.encoder.iter_blocks():
                block.register_parameter('prompt', None)
            if mode == 'T0':
                mae.requires_grad_(False)
            original.mode = mode
        if rate:
            for block, layer in enumerate(mae.encoder.stages[-1].blocks):
                dropout_paths += _insert_mlp_dropout(layer, f'mae.encoder.stages.3.blocks.{block}', rate)
        model = SelectiveClassifier(original, dropout_paths, rate)
        dropout_paths = list(model._dropout_paths)
    groups = _parameter_groups(model, mode)
    inventory = regularization_inventory(model)
    info = {'task_mode': mode,
         'arm': arm_id,
         'fixed_model': FIXED_MODEL.copy(),
         'initialization_seed': initial,
         'classification_seed': classification_seed if mode != 'MAE' else None,
         'source': source,
         'pretrained_state_loaded': pretrained_state is not None,
         'synthetic_mixer': _test_mixer_factory is not None or mechanisms_module is not None,
         'dropout_probability': rate,
         'dropout_paths': dropout_paths,
         'regularization_inventory': inventory,
         'gradient_groups': groups,
         'parameter_namespace': [{'name': name,
         'shape': list(parameter.shape),
         'elements': parameter.numel(),
         'requires_grad': parameter.requires_grad,
         'group': next((key for key,
         names in groups.items() if name in names))} for name,
         parameter in model.named_parameters()]}
    return (model, info)

def _shapes(value):
    if torch.is_tensor(value):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'device': str(value.device)}
    if isinstance(value, dict):
        return {key: _shapes(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_shapes(item) for item in value]
    return type(value).__name__

@contextmanager
def trace_module_calls(model):
    """Shape/call-count hook helper for an explicitly invoked real GPU G0 step."""
    records, handles = ({}, [])
    for name, module in model.named_modules():

        def hook(current, inputs, output, key=name):
            row = records.setdefault(key, {'class': type(current).__name__, 'calls': 0})
            row.update({'calls': row['calls'] + 1,
                 'last_input': _shapes(inputs),
                 'last_output': _shapes(output),
                 'training': current.training})
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
    grads = {name: None if parameter.grad is None else parameter.grad.detach().clone() for name,
         parameter in model.named_parameters()}
    rng = capture_rng()
    before = state_tensor_hash(model.state_dict())
    try:
        model.train()
        model.zero_grad(set_to_none=True)
        with trace_module_calls(model) as calls:
            if info['task_mode'] == 'MAE':
                output = model(images, pixel_mask=pixel_mask)
                selected = output['score_mask']
                difference = (output['prediction'] - output['target']).square()
                loss = torch.stack([difference[i].masked_select(selected[i]).mean() for i in range(images.shape[0])]).mean()
            else:
                if labels is None:
                    raise ValueError('Synthetic engineering classifier labels are required')
                output = model(images)
                loss = F.binary_cross_entropy_with_logits(output, labels.to(output))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite engineering data loss')
            loss.backward()
        stats = gradient_statistics(model, info['gradient_groups'])
        if not stats['global']['all_finite'] or stats['global']['frozen_with_grad_tensors']:
            raise ValueError('Nonfinite or frozen-parameter gradient in engineering probe')
        named = dict(model.named_parameters())
        disconnected = [group for group,
             names in info['gradient_groups'].items() if any((named[name].requires_grad for name in names)) and (not stats['groups'][group]['with_grad_elements'])]
        if disconnected:
            raise ValueError(f'Entire trainable groups disconnected from graph: {disconnected}')
        if state_tensor_hash(model.state_dict()) != before:
            raise ValueError('A forward/backward-only probe unexpectedly changed model state')
        return {'kind': 'ENGINEERING_FORWARD_BACKWARD_ONLY_NOT_RELEASE_CERTIFICATE',
             'synthetic_mixer': info['synthetic_mixer'],
             'device': str(images.device),
             'data_loss': float(loss.detach()),
             'output': _shapes(output),
             'module_calls': calls,
             'gradient_statistics': stats,
             'parameter_gradients': {name: {'requires_grad': parameter.requires_grad,
             'grad_present': parameter.grad is not None,
             'grad_l2': float(parameter.grad.detach().float().norm()) if parameter.grad is not None else None} for name,
             parameter in model.named_parameters()},
             'state_unchanged': True}
    finally:
        for name, parameter in model.named_parameters():
            parameter.grad = grads[name]
        for module, mode in modes:
            module.training = mode
        restore_rng(rng)
