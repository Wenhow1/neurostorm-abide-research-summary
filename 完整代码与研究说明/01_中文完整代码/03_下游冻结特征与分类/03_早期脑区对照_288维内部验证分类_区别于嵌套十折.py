# 早期966内部开发读出
# 作用：固定100轮的ROI模型，提取平均288维特征；分类器在同一内部验证集上选C和报告。
# 输入输出：约695人训练、174人验证（随折略有变化）；完整FC→[人,288]。
# 参数与配置：C只比较0.001/.003/.01，按内部验证BCE选取；属于探索性内部结果。
# 注意：此模块不能当成当前290人嵌套十折分类；它刻意保留历史评价差异。
# 来源模块：roi_block/readout.py
# 原始源码镜像：代码核验/原始源码/roi_block/readout.py
# 源文件SHA256：b7ff3e6971a083147a05e0516eb64e5de89b2e32967268826a2f6197a3c89eb5
# 完整性：本文件保留来源模块全部可执行语句，仅格式化并增加中文注释；Python AST逐项一致。
# 运行边界：这是保留原import名称的代码阅读版本。中文文件名不能替代原模块路径；运行需原数据、划分、权重、软件环境与原工程布局。未声称独立可运行。
# 副作用提醒：含main或顶层训练/提取语句的脚本若直接执行会读写文件或使用CUDA。本次整理未执行模型。

import pathlib, sys, os, json, time, argparse, hashlib

R = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(R))
import worker as w
import numpy as np, torch, joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from common import put, read, sha


def run(arm, f, seed, pilot=False):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = R / 'source'
    sp = read(source / 'splits.json')[f]
    tr = sp['inner_train']
    va = sp['inner_validation']
    ids = np.load(source / 'data/IDs.npy')
    raw = np.load(source / 'data/FC.npy', mmap_mode='r')
    y = np.load(source / 'data/Y.npy')
    name = f'{arm}_f{f}_s{seed}'
    out = R / ('pilot_readout' if pilot else 'readout') / name
    out.mkdir(exist_ok=True, parents=True)
    if (out / 'completion.json').exists():
        return
    if pilot:
        path = R / 'pilot' / f'{arm}_full/last.pt'
    elif arm == 'G0':
        path = (
            pathlib.Path(
                '/home/wenhao/neurostorm_2d/experiments/cpac966_decoder_mask_20261022/runs'
            )
            / f'B_f{f}_s{seed}/epoch_100.pt'
        )
    else:
        path = R / 'runs' / name / 'epoch_100.pt'
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck['epoch'] == (2 if pilot else 100)
    norm = ck['norm']
    assert norm['fit_subject_ids'] == ids[tr].tolist()
    m = w.make(f, seed, arm)
    m.load_state_dict(ck['model'])
    m.requires_grad_(False)
    m = m.cuda().eval()
    before = w.old.state_hash(m)
    calls = [0]
    hook = m.decoder.register_forward_hook(
        lambda *a: calls.__setitem__(0, calls[0] + 1)
    )
    q = tr + va
    features = []
    globals_ = []
    with torch.inference_mode():
        for k in range(0, len(q), 16):
            x = torch.from_numpy(
                ((raw[q[k : k + 16]] - norm['mean']) / norm['std']).astype('float32')
            )[:, None].cuda()
            z = m.encode_full(x, use_prompts=False)
            ch = 384 if arm == 'W' else 288
            assert tuple(z.shape[1:]) == (1, 116, ch)
            features.append(z.mean((1, 2)).cpu().numpy())
            globals_.append(z.mean((1, 2)).cpu().numpy())
    assert calls[0] == 0 and w.old.state_hash(m) == before
    hook.remove()
    del m, ck
    torch.cuda.empty_cache()
    x = np.concatenate(features)
    g = np.concatenate(globals_)
    assert np.isfinite(x).all()
    np.savez_compressed(out / 'features.npz', features=x, global_features=g, ids=ids[q])
    variants = ['global']
    result = {}
    for variant in variants:
        z = g if variant == 'global' else x
        scores = []
        models = []
        for c in [0.001, 0.003, 0.01]:
            steps = [StandardScaler()]
            if variant == 'pca256':
                steps.append(PCA(n_components=256, svd_solver='full'))
            steps.append(
                LogisticRegression(
                    C=c,
                    solver='liblinear',
                    max_iter=10000,
                    tol=1e-6,
                    random_state=2026102100 + f,
                )
            )
            model = make_pipeline(*steps)
            model.fit(z[: len(tr)], y[tr])
            assert model[0].n_samples_seen_ == len(tr)
            assert model[-1].n_iter_.max() < 10000
            pr = model.predict_proba(z[len(tr) :])[:, 1]
            scores.append(
                {
                    'C': c,
                    'train': w.old.metrics(
                        y[tr], model.predict_proba(z[: len(tr)])[:, 1]
                    ),
                    'validation': w.old.metrics(y[va], pr),
                }
            )
            models.append((model, pr))
        i = min(
            range(3), key=lambda i: (scores[i]['validation']['BCE'], scores[i]['C'])
        )
        model, pr = models[i]
        joblib.dump(model, out / f'{variant}_classifier.joblib')
        np.savez_compressed(
            out / f'{variant}_validation.npz', ids=ids[va], labels=y[va], probability=pr
        )
        result[variant] = {
            'selected': scores[i],
            'candidates': scores,
            'selection': 'inner validation BCE; exploratory reuse of validation',
        }
    put(
        out / 'completion.json',
        {
            'status': 'COMPLETE',
            'arm': arm,
            'fold': f,
            'seed': seed,
            'results': result,
            'feature_shape': list(x.shape),
            'checkpoint_sha256': sha(path),
            'feature_sha256': sha(out / 'features.npz'),
            'decoder_calls': 0,
            'backbone_unchanged': True,
            'outer_test_accessed': False,
            'time_unix': time.time(),
        },
    )


def summary():
    rows = []
    for arm in w.P['arms']:
        for seed in [1, 2]:
            for f in [0, 1, 2]:
                name = f'{arm}_f{f}_s{seed}'
                base = R / ('reuse' if arm == 'G0' else 'runs') / name
                rec = read(base / 'validation_mask50_subjects.json')
                rec75 = read(base / 'validation_mask75_subjects.json')
                rd = read(R / 'readout' / name / 'completion.json')
                assert rd['status'] == 'COMPLETE'
                row = {
                    'arm': arm,
                    'seed': seed,
                    'fold': f,
                    'MSE50': rec['MSE'],
                    'sym_MSE50': rec['sym_MSE'],
                    'MSE75': rec75['MSE'],
                    'residual_correlation': rec['residual_correlation'],
                    'residual_RMS_ratio': rec['residual_RMS_ratio'],
                    'amplitude_abs_error': float(
                        np.mean([abs(x['residual_RMS_ratio'] - 1) for x in rec['rows']])
                    ),
                }
                for k, v in rd['results'].items():
                    row.update(
                        {
                            k + '_' + m: v['selected']['validation'][m]
                            for m in ['AUROC', 'BCE', 'ACC']
                        }
                    )
                rows.append(row)
    import csv

    with (R / 'fold_results.csv').open('w') as f:
        a = csv.DictWriter(f, fieldnames=rows[0].keys())
        a.writeheader()
        a.writerows(rows)
    put(
        R / 'results.json',
        {
            'status': 'COMPLETE',
            'rows': rows,
            'baseline_reference': [
                x
                for x in read(R / 'baseline_reference.json')['rows']
                if x['arm'] == 'G0'
            ],
            'limitations': [
                'internal development only',
                'selected classifier on same inner validation; no independent confirmation',
                'six paired models not six independent cohorts',
            ],
            'time_unix': time.time(),
        },
    )


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('action', choices=['run', 'summary'])
    a.add_argument('--arm')
    a.add_argument('--fold', type=int, default=0)
    a.add_argument('--seed', type=int, default=1)
    a.add_argument('--pilot', action='store_true')
    args = a.parse_args()
    (
        summary()
        if args.action == 'summary'
        else run(args.arm, args.fold, args.seed, args.pilot)
    )
