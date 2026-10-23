# 共同遮挡条件的交叉评价
# 作用：分别加载方块训练和行列训练已保存权重，在完全相同的行列mask上比较。
# 输入输出：固定验证受试者；预测116×116；输出隐藏边MSE、个体残差相关和RMS幅度比。
# 参数与配置：50%与75%隐藏连接预算、3个固定mask视图；不更新权重。
# 注意：读代码可见while等待依赖完成；本次只阅读、没有运行或加载权重。
# 来源模块：roi_rows/cross_eval.py
# 原始源码镜像：代码核验/原始源码/roi_rows/cross_eval.py
# 源文件SHA256：569b0107e4ccf643178730fb41d5d1e4ae0ffd7c3fbd7cae37737cfbf40cf8db
# 完整性：本文件保留来源模块全部可执行语句，仅格式化并增加中文注释；Python AST逐项一致。
# 运行边界：这是保留原import名称的代码阅读版本。中文文件名不能替代原模块路径；运行需原数据、划分、权重、软件环境与原工程布局。未声称独立可运行。
# 副作用提醒：含main或顶层训练/提取语句的脚本若直接执行会读写文件或使用CUDA。本次整理未执行模型。

import sys, time, argparse, hashlib
from pathlib import Path

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
import worker as w
from common import read, put, sha
import numpy as np, torch

p = argparse.ArgumentParser()
p.add_argument('--fold', type=int, required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--pilot', action='store_true')
a = p.parse_args()
name = f'ROIR_f{a.fold}_s{a.seed}'
out = R / ('pilot_cross_eval' if a.pilot else 'cross_eval') / name
out.mkdir(exist_ok=True, parents=True)
if (out / 'completion.json').exists():
    sys.exit(0)
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
A = Path(read(R / 'protocol.json')['step1_reference'])
paths = {
    'step1_block': A / 'runs' / f'ROI_f{a.fold}_s{a.seed}' / 'epoch_100.pt',
    'step2_rows': R / 'runs' / name / 'epoch_100.pt',
}
if a.pilot:
    paths = {
        'step1_block': A / 'pilot/ROI_full/last.pt',
        'step2_rows': R / 'pilot/ROIR_full/last.pt',
    }
# First-step work is independent; never write into or restart its directory.
needed = list(paths.values())
if not a.pilot:
    needed.append(A / 'readout' / f'ROI_f{a.fold}_s{a.seed}' / 'completion.json')
while not all(q.is_file() for q in needed):
    time.sleep(3)
raw = np.load(R / 'source/data/FC.npy', mmap_mode='r')
ids = np.load(R / 'source/data/IDs.npy')
sp = read(R / 'source/splits.json')[a.fold]
tr = sp['inner_train']
va = sp['inner_validation']
prior = np.asarray(raw[tr]).mean(0)
summ = []
for label, path in paths.items():
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert (
        ck['epoch'] == (2 if a.pilot else 100)
        and ck['norm']['fit_subject_ids'] == ids[tr].tolist()
    )
    norm = ck['norm']
    m = w.make(a.fold, a.seed, 'ROIR').cuda().eval()
    m.load_state_dict(ck['model'])
    m.requires_grad_(False)
    before = w.old.state_hash(m)
    with torch.inference_mode():
        for ratio in [0.5, 0.75]:
            rows = []
            mh = hashlib.sha256()
            for view in range(3):
                for start in range(0, len(va), 8):
                    q = va[start : start + 8]
                    mask = w.row_masks(
                        m, ids[q], view, 'cpac966_roi_fixed_eval_v1', ratio
                    )
                    mh.update(mask.numpy().tobytes())
                    x = torch.from_numpy(
                        ((raw[q] - norm['mean']) / norm['std']).astype('float32')
                    )[:, None].cuda()
                    z = m(x, pixel_mask=mask.cuda())
                    score = z['score_mask'][:, 0].cpu().numpy()
                    pr = (
                        z['prediction'][:, 0].cpu().numpy() * norm['std'] + norm['mean']
                    )
                    for k, i in enumerate(q):
                        s = score[k]
                        t = raw[i][s]
                        p = pr[k][s]
                        r = t - prior[s]
                        v = p - prior[s]
                        rows.append(
                            {
                                'subject_id': str(ids[i]),
                                'view': view,
                                'MSE': float(np.mean((p - t) ** 2)),
                                'sym_MSE': float(
                                    np.mean((((pr[k] + pr[k].T) * 0.5)[s] - t) ** 2)
                                ),
                                'meanFC_MSE': float(np.mean(r**2)),
                                'residual_correlation': float(np.corrcoef(r, v)[0, 1]),
                                'residual_RMS_ratio': float(
                                    np.sqrt(np.mean(v**2) / np.mean(r**2))
                                ),
                                'score_fraction': float(s.sum() / (116 * 115)),
                            }
                        )
            rec = {
                'model': label,
                'mask': 'row_column',
                'ratio': ratio,
                'mask_sha256': mh.hexdigest(),
                'subjects': len(va),
                'views': 3,
                **{
                    k: float(np.mean([r[k] for r in rows]))
                    for k in [
                        'MSE',
                        'sym_MSE',
                        'meanFC_MSE',
                        'residual_correlation',
                        'residual_RMS_ratio',
                        'score_fraction',
                    ]
                },
            }
            put(out / f'{label}_rows{int(ratio*100)}.json', {**rec, 'rows': rows})
            summ.append(rec)
    assert w.old.state_hash(m) == before
    del m, ck
    torch.cuda.empty_cache()
for ratio in [0.5, 0.75]:
    assert len({x['mask_sha256'] for x in summ if x['ratio'] == ratio}) == 1
put(
    out / 'completion.json',
    {
        'status': 'COMPLETE',
        'rows': summ,
        'weights_unchanged': True,
        'outer_test_accessed': False,
        'no_training': True,
        'time': time.time(),
    },
)
