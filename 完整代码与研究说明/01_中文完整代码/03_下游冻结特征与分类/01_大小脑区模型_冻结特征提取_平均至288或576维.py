# 当前下游完整FC特征提取
# 作用：分别加载预训练与随机初始化的同结构编码器，保持权重冻结，平均脑区特征。
# 输入输出：输入290人完整FC；encode_full[B,1,116,d]；mean((1,2))→[B,d]；d为288或576。
# 参数与配置：上游norm来自610人；原始FC与上下游受试者不重叠检查；不调用重建头。
# 注意：这是已完成结果的GAP代码，新的flatten试验在本轮另立目录，不能把这里改名成flatten。
# 来源模块：schedule/features.py
# 原始源码镜像：代码核验/原始源码/schedule/features.py
# 源文件SHA256：0a205e3d3beb8c46dd4e04d6a08aea65327288543fad4f3f89a17f691751f8ee
# 完整性：本文件保留来源模块全部可执行语句，仅格式化并增加中文注释；Python AST逐项一致。
# 运行边界：这是保留原import名称的代码阅读版本。中文文件名不能替代原模块路径；运行需原数据、划分、权重、软件环境与原工程布局。未声称独立可运行。
# 副作用提醒：含main或顶层训练/提取语句的脚本若直接执行会读写文件或使用CUDA。本次整理未执行模型。

from pathlib import Path
import sys, json, hashlib, numpy as np, torch

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
import worker as w
from models import make
from common import put, sha

arm, seed = sys.argv[1], int(sys.argv[2])
root = R / 'refits' / f'{arm}_full610_s{seed}'
ck = torch.load(root / 'last.pt', map_location='cpu', weights_only=False)
assert (root / 'completion.json').exists()
norm = ck['norm']
assert len(norm['fit_subject_ids']) == 610
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
raw = np.load(R / 'downstream_data/FC.npy')
ids = np.load(R / 'downstream_data/IDs.npy')
assert not set(ids) & set(norm['fit_subject_ids'])
m = make(5, seed, arm).cuda()
out = R / 'features'
records = []
# 预训练和未预训练编码器都提取特征；后面的分类器仍需正常训练。
for kind, path in [('pretrained', root / 'last.pt'), ('random', root / 'initial.pt')]:
    c = torch.load(path, map_location='cpu', weights_only=False)
    m.load_state_dict(c['model'])
    m.eval()
    m.requires_grad_(False)
    before = w.old.state_hash(m)
    xs = []
    with torch.no_grad():
        for st in range(0, len(raw), 8):
            x = torch.from_numpy(
                ((raw[st : st + 8] - norm['mean']) / norm['std']).astype('float32')
            )[:, None].cuda()
            xs.append(m.encode_full(x).mean((1, 2)).cpu().numpy())
    assert before == w.old.state_hash(m)
    x = np.concatenate(xs)
    assert x.shape == (290, 576 if arm.endswith('L') else 288) and np.isfinite(x).all()
    name = f'{arm}_s{seed}_{kind}'
    np.save(out / f'{name}.npy', x)
    records.append(
        {
            'name': name,
            'shape': list(x.shape),
            'feature_sha256': sha(out / f'{name}.npy'),
            'checkpoint': str(path),
            'checkpoint_sha256': sha(path),
            'frozen_verified': True,
        }
    )
put(
    out / f'{arm}_s{seed}.json',
    {
        'status': 'COMPLETE',
        'records': records,
        'ids_sha256': sha(R / 'downstream_data/IDs.npy'),
    },
)
