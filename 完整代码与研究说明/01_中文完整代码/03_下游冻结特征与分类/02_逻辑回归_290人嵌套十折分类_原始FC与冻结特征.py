# 当前疾病分类全流程
# 作用：完整保留StandardScaler、L2逻辑回归、内部选C、外层测试、指标与逐折文件输出。
# 输入输出：特征[290,288]或[290,576]；R0为[290,6670]；输出290个外层测试概率。
# 参数与配置：外层10折261/29；内部3折；C为0.001/.003/.01/.03/.1/1；内层AUROC择优并以较小C打破平局；阈值0.5。
# 注意：没有隐藏层ReLU/GELU；逻辑回归输出概率对应Sigmoid。标准化每次仅拟合相应训练部分。
# 来源模块：schedule/classify.py
# 原始源码镜像：代码核验/原始源码/schedule/classify.py
# 源文件SHA256：8fc9450228d621b87e0431516fa07e71420998eeab8e11cbe54b77cc3da4fd26
# 完整性：本文件保留来源模块全部可执行语句，仅格式化并增加中文注释；Python AST逐项一致。
# 运行边界：这是保留原import名称的代码阅读版本。中文文件名不能替代原模块路径；运行需原数据、划分、权重、软件环境与原工程布局。未声称独立可运行。
# 副作用提醒：含main或顶层训练/提取语句的脚本若直接执行会读写文件或使用CUDA。本次整理未执行模型。

from pathlib import Path
import json, sys, time, hashlib, csv, warnings, numpy as np, joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    confusion_matrix,
)
from sklearn.exceptions import ConvergenceWarning

R = Path(__file__).resolve().parent


def put(p, x):
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(x, indent=2))
    t.replace(p)


def csvout(p, rows):
    with p.open('w') as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p >= 0.5, labels=[0, 1]).ravel()
    return {
        'AUROC': float(roc_auc_score(y, p)),
        'ACC': float(accuracy_score(y, p >= 0.5)),
        'BA': float(balanced_accuracy_score(y, p >= 0.5)),
        'BCE': float(log_loss(y, p, labels=[0, 1])),
        'sensitivity': float(tp / (tp + fn)),
        'specificity': float(tn / (tn + fp)),
    }


# 分类器定义：训练侧标准化后接L2逻辑回归；不含MLP隐藏层。
def pipe(C, seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            solver='liblinear',
            penalty='l2',
            max_iter=20000,
            tol=1e-6,
            random_state=seed,
        ),
    )


# 十折流程：只在外层训练部分的内部三折选择C，然后预测该外层测试折。
def run(name):
    warnings.filterwarnings('error', category=ConvergenceWarning)
    p = json.loads((R / 'protocol.json').read_text())
    splits = json.loads((R / 'downstream_splits.json').read_text())
    y = np.load(R / 'downstream_data/Y.npy')
    ids = np.load(R / 'downstream_data/IDs.npy')
    if name == 'R0':
        fc = np.load(R / 'downstream_data/FC.npy')
        u, v = np.triu_indices(116, 1)
        x = fc[:, u, v].astype('float64')
    else:
        receipt = json.loads(
            (R / 'features' / ('_'.join(name.split('_')[:2]) + '.json')).read_text()
        )
        r = next(r for r in receipt['records'] if r['name'] == name)
        assert (
            hashlib.sha256((R / 'features' / f'{name}.npy').read_bytes()).hexdigest()
            == r['feature_sha256']
        )
        x = np.load(R / 'features' / f'{name}.npy').astype('float64')
    out = R / 'downstream' / name
    out.mkdir(exist_ok=True)
    probs = np.full(290, np.nan)
    folds = np.full(290, -1)
    scores = []
    results = []
    for sp in splits:
        f = sp['fold']
        tr = np.array(sp['train'])
        te = np.array(sp['test'])
        choices = []
        for C in p['classifier']['C']:
            vals = []
            for k, ii in enumerate(sp['inner']):
                it = np.array(ii['train'])
                iv = np.array(ii['validation'])
                assert not (set(it) | set(iv)) & set(te)
                m = pipe(C, 2026102100 + f * 10 + k)
                m.fit(x[it], y[it])
                assert np.allclose(m[0].mean_, x[it].mean(0))
                pr = m.predict_proba(x[iv])[:, 1]
                auc = float(roc_auc_score(y[iv], pr))
                vals.append(auc)
                scores.append(
                    {
                        'fold': f,
                        'inner': k,
                        'C': C,
                        'AUROC': auc,
                        'train_n': len(it),
                        'validation_n': len(iv),
                    }
                )
            choices.append({'C': C, 'AUROC': float(np.mean(vals))})
        mx = max(c['AUROC'] for c in choices)
        C = min(c['C'] for c in choices if mx - c['AUROC'] <= 1e-12)
        put(
            out / f'selection_f{f}.json',
            {'C': C, 'inner': choices, 'outer_test_used': False},
        )
        m = pipe(C, 2026102190 + f)
        m.fit(x[tr], y[tr])
        pr = m.predict_proba(x[te])[:, 1]
        assert np.isnan(probs[te]).all()
        probs[te] = pr
        folds[te] = f
        results.append({'fold': f, 'C': C, **metrics(y[te], pr)})
        joblib.dump(m, out / f'classifier_f{f}.joblib')
        put(out / 'status.json', {'completed_folds': f + 1, 'time': time.time()})
    assert np.isfinite(probs).all()
    csvout(out / 'inner_scores.csv', scores)
    csvout(out / 'fold_results.csv', results)
    csvout(
        out / 'oof_predictions.csv',
        [
            {
                'subject_id': int(ids[i]),
                'fold': int(folds[i]),
                'label': int(y[i]),
                'probability': float(probs[i]),
            }
            for i in range(290)
        ],
    )
    put(
        out / 'completion.json',
        {
            'status': 'COMPLETE',
            'name': name,
            'subjects': 290,
            'nested_folds': '10x3',
            'fits': 190,
            'pooled': metrics(y, probs),
            'fold_equal_mean': {
                k: float(np.mean([r[k] for r in results])) for k in metrics(y, probs)
            },
            'time': time.time(),
        },
    )


if __name__ == '__main__':
    run(sys.argv[1])
