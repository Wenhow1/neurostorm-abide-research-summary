# 阅读副本：只展开压缩语句、调整空白并补充中文说明，函数与计算逻辑保持不变。
# 本文件不替换已经封定并正在运行的源码；原始文件、SHA256 与 AST 核验见同目录来源清单。
# 这不是单文件独立运行包：真实运行还需要原协议、旧受试者划分、数据、特征或检查点。
# 不要在阅读目录直接启动任务；部分路径与来源校验依赖原项目布局，__file__ 的位置也会改变。
# 任务：读取固定编码器已经保存的特征，比较逻辑回归、线性 SVM 与小型 MLP。
# 分类架构：冻结编码器的平均/展开特征 → 训练部分拟合标准化 → 分类器 → 疾病分数或概率。
# 本文件不调用编码器或重建解码器，不重新训练 MAE；只需要已提取特征、标签、ID 与既有划分。
# 数据：290 人，十折外层测试；每折261人训练/选参、29人测试，261人内部三折选择参数。
# 逻辑回归：L2、liblinear，C=[0.001,0.003,0.01,0.03,0.1,1]，无隐藏层；输出 Sigmoid 概率。
# 线性 SVM：L2、squared_hinge，C 同上；输出 decision 分数、阈值0，不伪造概率或 BCE。
# MLP：一个32单元隐藏层，ReLU；最后为二分类 logistic/Sigmoid 输出，阈值0.5。
# MLP：alpha=[0.001,0.01,0.1]，Adam初始学习率0.001，batch32，最多200轮。
# MLP：early_stopping=False；训练loss改善不足连续20轮可停止，不使用外层测试来选择轮次。
# 每个内层训练都单独拟合 StandardScaler；按内部平均AUROC选参数，平分时选更强正则化。
# C越小正则化越强，alpha越大正则化越强；每个外层折选完参数后再用261人重新拟合。
# 输出逐人外层测试预测、逐折成绩、所有候选分数、最终分类器、MLP训练曲线和收敛警告。
# 单进程单线程BLAS；只恢复带完整文件哈希的已完成外层折，失败不会伪装成完成。
# 原始文件：classifier.py
# 原始 SHA256：1c0191f874eeecef7e3ce089285b1c9f882e1ed32f3364f4fbe5de6987ba3b8f

"""Frozen-feature nested classification. No backbone training or test-driven selection."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
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


# 三个分类器的实际结构与超参数定义。
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


# 只用内部验证平均 AUROC 选参数，绝不读取外层测试来选。
def choose_candidate(choices, head):
    best = max((row['mean_inner_AUROC'] for row in choices))
    tied = [row for row in choices if best - row['mean_inner_AUROC'] <= 1e-12]
    return (max if normalize_head(head) == 'mlp' else min)(tied, key=lambda row: row['value'])


# SVM 保留无概率决策分数；逻辑回归和 MLP 使用二分类概率。
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


# 逐折检查训练/验证/测试没有交叉，每名受试者恰好被外层测试一次。
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


# 一次拟合：只在指定训练索引上拟合标准化与分类器，并保存训练诊断。
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


# 读取固定特征与旧划分并建立运行来源签名；不提取新特征。
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


# 已完成外层折只有文件和来源哈希全部吻合，才允许复用。
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


# 完整十折分类：内部选参 → 外层训练部分重新拟合 → 外层测试 → 保存与汇总。
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
if __name__ == '__main__':
    main()
