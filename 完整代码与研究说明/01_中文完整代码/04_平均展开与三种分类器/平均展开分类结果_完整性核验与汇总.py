# 阅读副本：只展开压缩语句、调整空白并补充中文说明，函数与计算逻辑保持不变。
# 本文件不替换已经封定并正在运行的源码；原始文件、SHA256 与 AST 核验见同目录来源清单。
# 这不是单文件独立运行包：真实运行还需要原协议、旧受试者划分、数据、特征或检查点。
# 不要在阅读目录直接启动任务；部分路径与来源校验依赖原项目布局，__file__ 的位置也会改变。
# 辅助任务：读取既有99项分类结果与8项配对特征提取记录，独立核验并生成短结果报告。
# 本文件不训练、不推理、不重新选分类器参数；需要原始结果目录和只读源标签/划分用于核验。
# 主跨分类器指标：十个外层折 AUROC 的等权平均；整体290人OOF AUROC作为辅助。
# 逐模型、逐种子、逐预训练状态、逐分类器，对同一划分的平均与展开结果配对比较。
# SVM 不提供概率，因此 BCE/Brier 必须缺失，不能把决策分数当概率。
# 默认只汇总核验通过的部分并明确PARTIAL；--strict要求99项分类和8项提取全通过。
# 完整性检查包括文件SHA256、十折覆盖、290人预测一次、标签与名单对应、参数选择及指标复算。
# 输出只反映已存在结果，不依据外层测试分数添加实验、删受试者或调整参数。
# 原始文件：summarize.py
# 原始 SHA256：d22d1c71b6c160d1c3535f5a4af9c7cb8aedc614e3d0f2897c2aeeeba2b86eda

"""Audit existing frozen-readout results and write a short Chinese summary.

Reads only. Does not fit models, choose new hyperparameters, or alter raw results.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss, roc_auc_score
ARMS = ['MS', 'TS', 'ML', 'TL']
SEEDS = [1, 2]
KINDS = ['pretrained', 'random']
READOUTS = ['gap', 'flatten']
HEADS = ['lr', 'svm', 'mlp']
NAMES = {'MS': '小 Mamba', 'TS': '小 Transformer', 'ML': '大 Mamba', 'TL': '大 Transformer', 'R0': '原始 FC'}
HEAD_NAMES = {'lr': '逻辑回归', 'svm': '线性 SVM', 'mlp': '单隐藏层 MLP'}
KIND_NAMES = {'pretrained': '经过重建预训练', 'random': '未做预训练', 'raw': '原始连接'}
READOUT_NAMES = {'gap': '所有脑区特征取平均', 'flatten': '保留全部脑区特征并展开', 'raw': '6670 条原始连接'}
REQUIRED_ARTIFACTS = {'selection.json',
    'inner_scores.csv',
    'fit_diagnostics.json',
    'result.json',
    'test_predictions.csv',
    'classifier.joblib'}

def read(path):
    with Path(path).open() as stream:
        return json.load(stream)

def put(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def csvout(path, rows, fields=None):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    if fields is None:
        fields = list(dict.fromkeys((k for r in rows for k in r)))
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ['status'])
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)

def resolved(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()

class HashCache:

    def __init__(self):
        self.cache = {}

    def __call__(self, path):
        path = Path(path).resolve()
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in self.cache:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda : stream.read(1024 * 1024), b''):
                    h.update(block)
            self.cache[key] = h.hexdigest()
        return self.cache[key]

def require(condition, text):
    if not condition:
        raise ValueError(text)

def near(actual, expected, label, atol=1e-10):
    if expected is None:
        require(actual is None, f'{label}: expected missing probability metric')
    else:
        require(actual is not None and math.isfinite(float(actual)) and (abs(float(actual) - expected) <= atol),
            f'{label}: reported value does not match recomputation ({actual} vs {expected})')

def scalar_metrics(y, scores, predicted, prob):
    (tn, fp, fn, tp) = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {'AUROC': float(roc_auc_score(y, scores)),
        'ACC': float(accuracy_score(y, predicted)),
        'BA': float(balanced_accuracy_score(y, predicted)),
        'BCE': None if prob is None else float(log_loss(y, prob, labels=[0, 1])),
        'Brier': None if prob is None else float(np.mean((prob - y) ** 2)),
        'sensitivity': float(tp / (tp + fn)),
        'specificity': float(tn / (tn + fp)),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp)}


# 预先固定的99项范围：四模型×两种子×两种初始化×两种读出×三头，再加原始FC三头。
def expected_jobs():
    out = []
    for head in HEADS:
        for readout in READOUTS:
            for arm in ARMS:
                for seed in SEEDS:
                    for kind in KINDS:
                        key = f'{arm}_s{seed}_{kind}_{readout}'
                        out.append({'name': key + '_' + head,
                            'feature_key': key,
                            'head': head,
                            'arm': arm,
                            'seed': seed,
                            'kind': kind,
                            'readout': readout})
    for head in HEADS:
        out.append({'name': 'R0_' + head,
            'feature_key': 'R0',
            'head': head,
            'arm': 'R0',
            'seed': 0,
            'kind': 'raw',
            'readout': 'raw'})
    return out


# 核对预先固定的任务、标签/ID/划分哈希和十折内部边界。
def get_context(root, digest):
    p = read(root / 'protocol.json')
    manifest = read(root / 'jobs.json')
    expected = expected_jobs()
    mapping = {j['name']: j for j in expected}
    require(len(manifest) == 99 and len({j['name'] for j in manifest}) == 99,
        'jobs.json must contain 99 distinct planned tasks')
    require({j['name'] for j in manifest} == set(mapping), 'Job names differ from fixed 99-task design')
    for j in manifest:
        require(all((j.get(k) == v for (k, v) in mapping[j['name']].items())),
            f"Manifest fields disagree for {j['name']}")
        actual = read(root / 'jobs' / (j['name'] + '.json'))
        require(all((actual.get(k) == v for (k, v) in mapping[j['name']].items())),
            'Per-job file differs from jobs.json')
    data = p['data']
    source = resolved(root, data['source_root'])
    paths = {k: resolved(root,
        data.get(k + '_path', str(source / tail))) for (k,
        tail) in [('labels', 'downstream_data/Y.npy'),
        ('ids', 'downstream_data/IDs.npy'),
        ('splits', 'downstream_splits.json')]}
    for (key, path) in paths.items():
        require(digest(path) == p['input_hashes'][key],
            f'Source {key} hash differs from frozen protocol')
    y = np.load(paths['labels'], allow_pickle=False)
    ids = np.load(paths['ids'], allow_pickle=False)
    splits = read(paths['splits'])
    require(y.shape == (290,) and ids.shape == (290,), 'Expected 290 aligned subjects')
    require(set(np.unique(y)) == {0, 1} and len(set(map(str, ids))) == 290,
        'Invalid labels or duplicate IDs')
    require(len(splits) == 10 and {s['fold'] for s in splits} == set(range(10)), 'Expected folds 0..9')
    all_indices = set(range(290))
    tested = []
    for sp in splits:
        (tr, te) = (set(sp['train']), set(sp['test']))
        require(len(tr) == 261 and len(te) == 29 and (not tr & te) and (tr | te == all_indices),
            f"Outer split invalid: {sp['fold']}")
        require(len(sp['train']) == 261 and len(sp['test']) == 29, 'Duplicate outer indices')
        require(len(sp['inner']) == 3, 'Expected three inner folds')
        inner_tested = []
        for inner in sp['inner']:
            (it, iv) = (set(inner['train']), set(inner['validation']))
            require(not it & iv and it | iv == tr and (not (it | iv) & te), 'Inner/outer split leakage')
            require(len(it) == len(inner['train']) and len(iv) == len(inner['validation']),
                'Duplicate inner indices')
            inner_tested.extend(inner['validation'])
        require(sorted(inner_tested) == sorted(tr), 'Inner validation is not once-only coverage')
        tested.extend(sp['test'])
    require(sorted(tested) == list(range(290)), 'Outer testing is not once-only coverage')
    return (p, manifest, paths, y.astype(int), ids, {s['fold']: s for s in splits})


# 核验平均/展开特征来自同一冻结编码器、同一名单与相同脑区顺序。
def check_receipts(root, p, ids, digest):
    (audits, records) = ([], {})
    for arm in ARMS:
        for seed in SEEDS:
            name = f'{arm}_s{seed}'
            path = root / 'receipts' / (name + '.json')
            if not path.exists():
                audits.append({'name': name, 'status': 'PENDING', 'reason': '尚无完整特征提取记录'})
                continue
            try:
                rc = read(path)
                d = 576 if arm in ('ML', 'TL') else 288
                require(rc.get('status') == 'COMPLETE' and rc.get('subjects') == 290,
                    'Receipt not complete for 290 subjects')
                require(rc.get('ids_sha256') == p['input_hashes']['ids'], 'Receipt ID hash mismatch')
                require(rc.get('subject_ids_in_output_order') == list(map(str, ids)),
                    'Feature row order disagrees with source IDs')
                require(rc.get('no_training') is True and rc.get('backward_called') is False,
                    'Receipt does not confirm frozen inference')
                require(rc.get('decoder_calls') == 0 and rc.get('upstream_holdout_accessed') is False,
                    'Unexpected decoder or upstream holdout access')
                require(rc.get('provenance', {}).get('protocol_sha256') == digest(root / 'protocol.json'),
                    'Extraction protocol hash mismatch')
                expected = {f'{name}_{kind}_{readout}' for kind in KINDS for readout in READOUTS}
                require(len(rc['records']) == 4 and {r['name'] for r in rc['records']} == expected,
                    'Receipt requires all four matched pretrained/random GAP/flatten matrices')
                require(len(rc['checks']) == 2 and {v['kind'] for v in rc['checks']} == set(KINDS),
                    'Missing paired checkpoint checks')
                for check in rc['checks']:
                    require(check.get('frozen_verified') is True and check.get('decoder_calls') == 0 and (check.get('model_state_before') == check.get('model_state_after')),
                        'Frozen-state check failed')
                verified = {}
                for record in rc['records']:
                    feature_path = resolved(root, record['path'])
                    require(digest(feature_path) == record['feature_sha256'],
                        'Feature checksum mismatch: ' + record['name'])
                    require(record['ids_sha256'] == p['input_hashes']['ids'],
                        'Feature ID digest mismatch')
                    shape = [290, d if record['readout'] == 'gap' else 116 * d]
                    require(record['shape'] == shape, 'Declared feature shape mismatch')
                    matrix = np.load(feature_path, mmap_mode='r', allow_pickle=False)
                    require(list(matrix.shape) == shape, 'Actual feature shape mismatch')
                    del matrix
                    verified[record['name']] = record
                records.update(verified)
                audits.append({'name': name,
                    'status': 'VERIFIED',
                    'files': 4,
                    'receipt_sha256': digest(path)})
            except Exception as exc:
                audits.append({'name': name, 'status': 'INVALID', 'reason': str(exc)})
    return (audits, records)


# 从逐人预测重新计算指标，检查概率与阈值的语义。
def check_predictions(rows, expected_indices, ids, y, head, fold):
    require(len(rows) == len(expected_indices), 'Prediction count mismatch')
    require([int(r['index']) for r in rows] == list(expected_indices), 'Prediction index order mismatch')
    for (row, i) in zip(rows, expected_indices):
        require(str(row['subject_id']) == str(ids[i]) and int(row['label']) == int(y[i]) and (int(row['fold']) == fold),
            'Prediction subject/label/fold mismatch')
        score = float(row['score'])
        require(math.isfinite(score), 'Nonfinite prediction score')
        if head == 'svm':
            require(row.get('probability') is None or row.get('probability') == '',
                'SVM must not invent probability')
            require(int(row['predicted_label']) == int(score >= 0), 'SVM threshold must be zero')
        else:
            prob = float(row['probability'])
            require(math.isfinite(prob) and 0 <= prob <= 1, 'Invalid probability')
            require(abs(prob - score) <= 1e-12 and int(row['predicted_label']) == int(prob >= 0.5),
                'Probability/threshold mismatch')
    score = np.array([float(r['score']) for r in rows])
    pred = np.array([int(r['predicted_label']) for r in rows])
    prob = None if head == 'svm' else np.array([float(r['probability']) for r in rows])
    return scalar_metrics(y[np.asarray(expected_indices)], score, pred, prob)


# 独立检查一个完整十折分类任务；损坏、缺项或不一致不计为完成。
def audit_job(root, p, paths, job, y, ids, splits, records, digest):
    out = resolved(root, p.get('output_dir', 'downstream')) / job['name']
    completed = out / 'completion.json'
    if not completed.exists():
        status = read(out / 'status.json') if (out / 'status.json').exists() else {}
        failures = list(out.glob('fold_*/failure.json'))
        return ({'name': job['name'], 'status': 'FAILED' if status.get('status') == 'FAILED' or failures else 'PENDING', 'reason': status.get('error', '尚未完成全部十个外层折'), 'completed_folds_unverified': len(list(out.glob('fold_*/completion.json')))},
            None,
            [])
    try:
        completed_hash = digest(completed)
        complete = read(completed)
        config = read(out / 'config.json')
        require(complete.get('status') == 'COMPLETE' and complete.get('subjects') == 290 and (complete.get('nested_folds') == '10x3'),
            'Task completion contract mismatch')
        require(complete.get('head') == job['head'] and complete.get('name') == job['name'],
            'Completed task identity mismatch')
        require(complete.get('probability_available') is (job['head'] != 'svm'),
            'Task probability contract mismatch')
        require(all((config.get('job', {}).get(k) == v for (k, v) in job.items())),
            'Saved job config mismatch')
        sigmat = config['provenance']
        calculated_sig = hashlib.sha256(json.dumps(sigmat, sort_keys=True).encode()).hexdigest()
        require(config['signature'] == complete['signature'] == calculated_sig,
            'Task signature mismatch')
        require(sigmat['protocol_sha256'] == digest(root / 'protocol.json'),
            'Classifier protocol digest mismatch')
        require(sigmat['code_sha256'] == digest(root / 'classifier.py'),
            'Classifier source digest mismatch')
        for key in ('labels', 'ids', 'splits'):
            require(sigmat['inputs'][key] == digest(paths[key]), f'Classifier {key} digest mismatch')
        feature = resolved(root, p.get('feature_dir', 'features')) / (job['feature_key'] + '.npy')
        require(digest(feature) == sigmat['inputs']['features'], 'Classifier feature digest mismatch')
        if job['arm'] != 'R0':
            extraction = root / 'receipts' / f"{job['arm']}_s{job['seed']}.json"
            if not extraction.exists():
                return ({'name': job['name'], 'status': 'PENDING', 'reason': '分类已完成，等待对应特征提取记录核验后纳入汇总'},
                    None,
                    [])
            require(job['feature_key'] in records, 'Matched extraction receipt not yet verified')
            require(records[job['feature_key']]['feature_sha256'] == sigmat['inputs']['features'],
                'Receipt/classifier feature mismatch')
        folds = []
        all_rows = []
        warning_count = 0
        require({q.name for q in out.glob('fold_*/completion.json')} == {'completion.json'},
            'No fold completions')
        actual_fold_dirs = {q.parent.name for q in out.glob('fold_*/completion.json')}
        require(actual_fold_dirs == {f'fold_{i:02d}' for i in range(10)},
            'Require exactly ten completed outer folds')
        for f in range(10):
            directory = out / f'fold_{f:02d}'
            rc = read(directory / 'completion.json')
            require(rc.get('status') == 'COMPLETE' and rc.get('signature') == calculated_sig,
                f'Fold {f} completion signature mismatch')
            require(set(rc['artifact_sha256']) == REQUIRED_ARTIFACTS,
                f'Fold {f} artifact inventory mismatch')
            for (filename, expected) in rc['artifact_sha256'].items():
                require(digest(directory / filename) == expected,
                    f'Fold {f} checksum mismatch: {filename}')
            result = read(directory / 'result.json')
            selection = read(directory / 'selection.json')
            require(result['fold'] == f and selection['fold'] == f and (selection.get('outer_test_used') is False),
                f'Fold {f} selection contract mismatch')
            param = 'alpha' if job['head'] == 'mlp' else 'C'
            grid = [0.001, 0.01, 0.1] if job['head'] == 'mlp' else [0.001, 0.003, 0.01, 0.03, 0.1, 1.0]
            choices = selection['candidates']
            require(sorted((c['value'] for c in choices)) == grid, 'Candidate grid mismatch')
            for candidate in choices:
                require(len(candidate['inner_AUROC']) == 3,
                    'Require three inner scores for every candidate')
                near(candidate['mean_inner_AUROC'],
                    float(np.mean(candidate['inner_AUROC'])),
                    'Inner candidate mean')
            maximum = max((v['mean_inner_AUROC'] for v in choices))
            tied = [v for v in choices if maximum - v['mean_inner_AUROC'] <= 1e-12]
            chosen = (max if job['head'] == 'mlp' else min)(tied, key=lambda v: v['value'])
            require(selection['chosen'] == chosen and result['selected_parameter'] == param and (result['selected_value'] == chosen['value']),
                'Inner-only parameter selection mismatch')
            measured = check_predictions(result['predictions'],
                splits[f]['test'],
                ids,
                y,
                job['head'],
                f)
            for (k, v) in measured.items():
                near(result['test_metrics'][k], v, f'Fold {f}/{k}')
            with (directory / 'test_predictions.csv').open() as stream:
                csv_rows = list(csv.DictReader(stream))
            check_predictions(csv_rows, splits[f]['test'], ids, y, job['head'], f)
            for (a, b) in zip(csv_rows, result['predictions']):
                near(float(a['score']), float(b['score']), 'Prediction CSV/JSON score')
            diagnostics = read(directory / 'fit_diagnostics.json')
            require(len(diagnostics) == len(grid) * 3 + 1, 'Fit diagnostic count mismatch')
            for diagnostic in diagnostics:
                require(diagnostic.get('scaler_fit_only_train') is True,
                    'Missing train-only scaler confirmation')
                if job['head'] == 'mlp':
                    require(diagnostic.get('output_activation') == 'logistic' and diagnostic.get('no_validation_early_stopping') is True,
                        'MLP output/stop contract mismatch')
                    require(len(diagnostic['loss_curve']) == diagnostic['n_iter'],
                        'MLP loss history incomplete')
            warnings = sum((bool(v.get('convergence_warning')) for v in diagnostics))
            if job['head'] != 'mlp':
                require(warnings == 0, 'Unconverged linear classifier must not count as complete')
            require(warnings == result['convergence_warning_fits'], 'Convergence-warning count mismatch')
            warning_count += warnings
            fold = {**job,
                'fold': f,
                'selected_parameter': param,
                'selected_value': chosen['value'],
                'inner_selected_AUROC': chosen['mean_inner_AUROC'],
                **measured,
                'train_AUROC': result['train_metrics']['AUROC'],
                'train_BCE': result['train_metrics']['BCE'],
                'convergence_warning_fits': warnings}
            folds.append(fold)
            all_rows.extend(result['predictions'])
        all_rows.sort(key=lambda r: r['index'])
        require([r['index'] for r in all_rows] == list(range(290)),
            'OOF coverage incomplete or duplicated')
        with (out / 'oof_predictions.csv').open() as stream:
            stored = list(csv.DictReader(stream))
        require(len(stored) == 290, 'OOF CSV incomplete')
        for (a, b) in zip(stored, all_rows):
            for k in ('index', 'fold', 'label', 'predicted_label'):
                require(int(a[k]) == b[k], 'OOF CSV identity mismatch')
            require(a['subject_id'] == b['subject_id'], 'OOF CSV subject mismatch')
            near(float(a['score']), b['score'], 'OOF CSV score')
            if job['head'] == 'svm':
                require(a['probability'] == '', 'SVM OOF CSV contains invented probabilities')
            else:
                near(float(a['probability']), b['probability'], 'OOF CSV probability')
        scores = np.array([r['score'] for r in all_rows])
        predicted = np.array([r['predicted_label'] for r in all_rows])
        probs = None if job['head'] == 'svm' else np.array([r['probability'] for r in all_rows])
        pooled = scalar_metrics(y, scores, predicted, probs)
        for (k, v) in pooled.items():
            near(complete['pooled'][k], v, 'Pooled ' + k)
        means = {k: None if pooled[k] is None else float(np.mean([v[k] for v in folds])) for k in ('AUROC',
            'ACC',
            'BA',
            'BCE',
            'Brier',
            'sensitivity',
            'specificity')}
        for (k, v) in means.items():
            near(complete['fold_equal_mean'][k], v, 'Fold mean ' + k)
        require(complete['convergence_warning_fits'] == warning_count,
            'Task convergence-warning count mismatch')
        require(digest(completed) == completed_hash, 'Task changed while auditing; retry snapshot')
        row = {**job,
            'status': 'VERIFIED',
            'subjects': 290,
            'completed_outer_folds': 10,
            'primary_fold_mean_AUROC': means['AUROC'],
            'auxiliary_pooled_AUROC': pooled['AUROC'],
            'ACC': pooled['ACC'],
            'BA': pooled['BA'],
            'BCE': pooled['BCE'],
            'Brier': pooled['Brier'],
            'probability_available': job['head'] != 'svm',
            'convergence_warning_fits': warning_count,
            'completion_sha256': completed_hash}
        return ({'name': job['name'], 'status': 'VERIFIED', 'folds': 10, 'subjects': 290}, row, folds)
    except Exception as exc:
        return ({'name': job['name'], 'status': 'INVALID', 'reason': str(exc)}, None, [])


# 只配对同模型、同种子、同初始化和同分类器，计算展开减平均的差。
def paired_readout(rows, folds):
    lookup = {(r['arm'], r['seed'], r['kind'], r['head'], r['readout']): r for r in rows}
    fl = {(r['name'], r['fold']): r for r in folds}
    paired = []
    for arm in ARMS:
        for seed in SEEDS:
            for kind in KINDS:
                for head in HEADS:
                    base = (arm, seed, kind, head)
                    if (*base, 'gap') not in lookup or (*base, 'flatten') not in lookup:
                        continue
                    (a, b) = (lookup[(*base, 'gap')], lookup[(*base, 'flatten')])
                    for f in range(10):
                        (af, bf) = (fl[a['name'], f], fl[b['name'], f])
                        paired.append({'level': 'outer_fold',
                            'arm': arm,
                            'seed': seed,
                            'kind': kind,
                            'head': head,
                            'fold': f,
                            'gap_AUROC': af['AUROC'],
                            'flatten_AUROC': bf['AUROC'],
                            'delta_AUROC_flatten_minus_gap': bf['AUROC'] - af['AUROC'],
                            'gap_BCE': af['BCE'],
                            'flatten_BCE': bf['BCE'],
                            'delta_BCE_flatten_minus_gap': None if head == 'svm' else bf['BCE'] - af['BCE']})
                    paired.append({'level': 'ten_fold_mean',
                        'arm': arm,
                        'seed': seed,
                        'kind': kind,
                        'head': head,
                        'fold': '',
                        'gap_AUROC': a['primary_fold_mean_AUROC'],
                        'flatten_AUROC': b['primary_fold_mean_AUROC'],
                        'delta_AUROC_flatten_minus_gap': b['primary_fold_mean_AUROC'] - a['primary_fold_mean_AUROC'],
                        'gap_BCE': a['BCE'],
                        'flatten_BCE': b['BCE'],
                        'delta_BCE_flatten_minus_gap': None if head == 'svm' else b['BCE'] - a['BCE']})
    return paired

def fmt(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


# 生成中文短结果；未完成时明确标记，不能提前宣称方法获胜。
def build_report(audit, rows, folds, paired):
    complete = audit['status'] == 'COMPLETE'
    lookup = {r['name']: r for r in rows}
    lines = ['# 固定编码器后的特征展开与分类器对照',
        '',
        f"记录时间：{audit['time']}。状态：**{audit['status']}**。已独立核验 {len(rows)}/99 项分类任务、{audit['features_verified']}/8 项成对特征提取。",
        '',
        '**目前只分析核验通过的既有结果；未完成、失败或校验异常的任务不混入成绩。**' if not complete else '**99 项分类任务与 8 项成对特征提取均已通过完整性核验。**',
        '',
        '## 本次回答什么',
        '',
        '1. 保持编码器、受试者划分和分类器一致，把所有脑区取平均改为完整展开，分类是否更好？',
        '2. 在相同特征上，把逻辑回归换成线性 SVM 或单隐藏层 MLP，分类是否更好？',
        '3. 经过重建预训练的编码器，是否优于同结构但未预训练的编码器？',
        '',
        '编码器保持固定，不重新训练 MAE。610 人用于此前上游重建预训练；本次仍在另一批 290 人上使用既定十折外层测试与三折内部选参。此人群此前已经检查过，本轮是继续开发的对照，不能称为全新独立外部验证。',
        '',
        '## 成绩怎样读',
        '',
        '**跨分类器的主要比较指标是十个外层折 AUROC 的等权平均。** 每个预训练种子分别报告；两个种子平均只在二者都核验完成时计算。整体 290 人 OOF AUROC 是辅助指标，尤其 SVM 的不同折决策分数没有统一概率校准。',
        '',
        'BCE 和 Brier 是概率误差，仅报告逻辑回归和 MLP。线性 SVM 提供决策分数，既不伪造概率，也不报告 BCE。分类器参数只根据内部验证选择；本汇总不按测试成绩再选参数。',
        '',
        '## 四种模型的成绩',
        '',
        '|模型|编码器状态|特征提供方式|分类器|第1次训练 AUROC|第2次训练 AUROC|两次平均|完成数|',
        '|---|---|---|---|---:|---:|---:|---:|']
    for arm in ARMS:
        for kind in KINDS:
            for readout in READOUTS:
                for head in HEADS:
                    matched = [lookup.get(f'{arm}_s{s}_{kind}_{readout}_{head}') for s in SEEDS]
                    available = [r for r in matched if r is not None]
                    if not available:
                        continue
                    vals = [r['primary_fold_mean_AUROC'] if r else None for r in matched]
                    avg = float(np.mean(vals)) if all((v is not None for v in vals)) else None
                    lines.append(f'|{NAMES[arm]}|{KIND_NAMES[kind]}|{READOUT_NAMES[readout]}|{HEAD_NAMES[head]}|{fmt(vals[0])}|{fmt(vals[1])}|{fmt(avg)}|{len(available)}/2|')
    if not rows:
        lines.append('|尚无核验通过结果||||||||')
    lines += ['',
        '## 原始 FC 参照',
        '',
        '|分类器|十折平均 AUROC（主要）|整体 OOF AUROC（辅助）|准确率|BCE|',
        '|---|---:|---:|---:|---:|']
    for head in HEADS:
        r = lookup.get('R0_' + head)
        if r:
            lines.append(f"|{HEAD_NAMES[head]}|{fmt(r['primary_fold_mean_AUROC'])}|{fmt(r['auxiliary_pooled_AUROC'])}|{r['ACC']:.2%}|{fmt(r['BCE'])}|")
        else:
            lines.append(f'|{HEAD_NAMES[head]}|待完成|—|—|—|')
    lines += ['',
        '## 平均改为展开：同一个分类器的配对变化',
        '',
        '下表正值表示展开后的十折平均 AUROC 较高；逐折数值见 paired_readout.csv。两次训练方向一致仅是稳定性线索，不等于已完成统计显著性检验。',
        '',
        '|模型|编码器状态|分类器|第1次训练变化|第2次训练变化|两次方向|',
        '|---|---|---|---:|---:|---|']
    paired_lookup = {(r['arm'],
        r['seed'],
        r['kind'],
        r['head']): r for r in paired if r['level'] == 'ten_fold_mean'}
    lr_improved = []
    for arm in ARMS:
        for kind in KINDS:
            for head in HEADS:
                pair = [paired_lookup.get((arm, s, kind, head)) for s in SEEDS]
                if not any(pair):
                    continue
                deltas = [r['delta_AUROC_flatten_minus_gap'] if r else None for r in pair]
                direction = '尚未配齐两次训练'
                if all((v is not None for v in deltas)):
                    direction = '两次均提高' if all((v > 0 for v in deltas)) else '两次均降低' if all((v < 0 for v in deltas)) else '方向不一致或持平'
                    if kind == 'pretrained' and head == 'lr':
                        lr_improved.append((arm, deltas, direction))
                lines.append(f'|{NAMES[arm]}|{KIND_NAMES[kind]}|{HEAD_NAMES[head]}|{fmt(deltas[0])}|{fmt(deltas[1])}|{direction}|')
    if not paired:
        lines.append('|尚无完整平均/展开配对||||||')
    lines += ['', '## 当前能够说明的事', '']
    if not complete:
        lines.append('- 当前是部分结果，不能据此宣布某个模型、读出方式或分类器获胜。')
    if lr_improved:
        for (arm, deltas, direction) in lr_improved:
            lines.append(f'- 在固定逻辑回归下，{NAMES[arm]}的展开减平均 AUROC 为 {deltas[0]:+.4f}、{deltas[1]:+.4f}，{direction}。这是读出方式的配对观察。')
    else:
        lines.append('- 预训练模型的逻辑回归平均/展开对照尚未配齐两个种子，暂不判断平均池化是否影响分类。')
    warnings = sum((r['convergence_warning_fits'] for r in rows))
    lines.append(f'- 已核验任务中共有 {warnings} 次拟合记录了收敛警告；逐任务见 comparison.csv，逐次训练曲线与警告见原始 fit_diagnostics.json。MLP 达到训练预算后的警告不被隐去，仍需结合训练曲线解释。')
    lines += ['- 展开特征保留编码器输出的脑区差异，不能恢复编码器此前已丢失的信息。小模型展开为 33,408 维，大模型为 66,816 维；平均后分别为 288、576 维。',
        '- MLP 的输入层参数随输入维度增加，因此“展开加 MLP”的变化不能全部归因于池化；先看保持同一逻辑回归的配对对照。',
        '- 三种分类器的内层搜索预算不相同：逻辑回归和 SVM 搜索六个 C，MLP 搜索三个 alpha；这是明确限定的候选对照，不是无限调参后的最优分类器比较。',
        '- 这些外层折和两个种子共用受试者，不能当成互不相关的独立样本；本短报告不提供未经设计的显著性宣称。',
        '',
        '## 未完成或异常',
        '']
    problematic = [r for r in audit['tasks'] if r['status'] != 'VERIFIED']
    receipt_problems = [r for r in audit['feature_receipts'] if r['status'] != 'VERIFIED']
    if not problematic and (not receipt_problems):
        lines.append('全部预定任务均已核验，没有遗漏。')
    else:
        lines += ['|对象|状态|原因|', '|---|---|---|']
        for r in receipt_problems + problematic:
            lines.append(f"|{r['name']}|{r['status']}|{str(r.get('reason', '')).replace('|', '/').replace(chr(10), ' ')}|")
    lines += ['',
        '## 文件入口',
        '',
        '- comparison.csv：逐模型、逐种子、逐编码器状态、逐读出与分类器的成绩。',
        '- fold_details.csv：每项任务的十个外层折成绩、内部选中参数及训练表现。',
        '- paired_readout.csv：同模型/种子/状态/分类器的平均与展开，逐折及十折均值配对差。',
        '- verification.json：逐任务和特征提取的完整性核验；失败或未完成不会被计为完成。',
        '- artifacts_sha256.json：本次汇总文件校验值。',
        '']
    return '\n'.join(lines)


# 汇总入口：完成状态取决于99项分类和8项提取全部核验，严格模式不满足即失败。
def summarize(root, strict=False):
    root = Path(root).resolve()
    digest = HashCache()
    output = root / 'results'
    output.mkdir(parents=True, exist_ok=True)
    now = datetime.now(ZoneInfo('Asia/Singapore')).isoformat(timespec='seconds')
    try:
        (p, jobs, paths, y, ids, splits) = get_context(root, digest)
        (receipt_audits, records) = check_receipts(root, p, ids, digest)
        audits = []
        rows = []
        folds = []
        for job in jobs:
            (audit,
                row,
                individual_folds) = audit_job(root,
                p,
                paths,
                job,
                y,
                ids,
                splits,
                records,
                digest)
            audits.append(audit)
            if row is not None:
                rows.append(row)
                folds.extend(individual_folds)
        features_verified = sum((a['status'] == 'VERIFIED' for a in receipt_audits))
        complete = len(rows) == 99 and features_verified == 8
        audit = {'status': 'COMPLETE' if complete else 'PARTIAL',
            'time': now,
            'classifiers_verified': len(rows),
            'classifiers_planned': 99,
            'features_verified': features_verified,
            'features_planned': 8,
            'invalid_or_failed_tasks': sum((a['status'] in ('FAILED', 'INVALID') for a in audits)),
            'invalid_feature_receipts': sum((a['status'] == 'INVALID' for a in receipt_audits)),
            'tasks': audits,
            'feature_receipts': receipt_audits,
            'protocol_sha256': digest(root / 'protocol.json'),
            'summarizer_sha256': digest(__file__),
            'strict_gate_passed': complete,
            'primary_metric': 'equal-weight mean outer-fold AUROC',
            'auxiliary_metric': 'pooled OOF AUROC',
            'test_based_parameter_selection': False}
        pairs = paired_readout(rows, folds)
        csvout(output / 'comparison.csv', rows)
        csvout(output / 'fold_details.csv', folds)
        csvout(output / 'paired_readout.csv', pairs)
        put(output / 'verification.json', audit)
        (output / 'report.md.tmp').write_text(build_report(audit, rows, folds, pairs))
        (output / 'report.md.tmp').replace(output / 'report.md')
        filenames = ['comparison.csv',
            'fold_details.csv',
            'paired_readout.csv',
            'verification.json',
            'report.md']
        put(output / 'artifacts_sha256.json', {n: digest(output / n) for n in filenames})
        return (audit, 0 if not strict or complete else 2)
    except Exception as error:
        audit = {'status': 'AUDIT_FAILED',
            'time': now,
            'error': str(error),
            'strict_gate_passed': False,
            'classifiers_planned': 99,
            'features_planned': 8}
        put(output / 'verification.json', audit)
        (output / 'report.md').write_text('# 固定编码器后的特征展开与分类器对照\n\n**完整性检查未通过，不能将本目录的成绩视为已核验结果。**\n\n原因：' + str(error) + '\n')
        for name in ('comparison.csv', 'fold_details.csv', 'paired_readout.csv'):
            csvout(output / name, [], ['status'])
        put(output / 'artifacts_sha256.json',
            {n: digest(output / n) for n in ['comparison.csv', 'fold_details.csv', 'paired_readout.csv', 'verification.json', 'report.md']})
        return (audit, 2)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent))
    parser.add_argument('--strict',
        action='store_true',
        help='Exit 2 unless all 99 classifier tasks and eight feature receipts verify')
    args = parser.parse_args()
    (audit, code) = summarize(args.root, args.strict)
    print(json.dumps({k: v for (k, v) in audit.items() if k not in ('tasks', 'feature_receipts')}, ensure_ascii=False),
        flush=True)
    return code
if __name__ == '__main__':
    sys.exit(main())
