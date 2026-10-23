# 09_旧小块Mamba_G1噪声与G2时序窗口_实际增强模块
# 归档主题：主题03
# 原增强模块完整代码，直接看到哪些输入改变、哪些重建目标改变。
# 输入/输出：输入与目标均每人116×116 FC；对称性保留。G2使用预先生成的窗口FC文件。
# 本实验改动：每人每轮50%概率增强。G1只在可见上三角边加标准差0.02高斯噪声并对称回填、截断[-1,1]，目标仍原FC；G2从已存窗口中选一个，同时替换输入与目标。
# 参数在哪里：arrays()、rng_for()和selected()；窗口长度与覆盖约束应看同目录窗口准备代码和protocol.json，不能从本模块推定。
# 证据范围：增强仅用于上游训练批次；固定验证使用原FC；不是把验证噪声当数据增强。
# 源文件：03_数据增强与编码器结构/cpac966_augmentation_20261022/augment.py
# 原始字节 SHA256：188e58ebd20bcab603123c57b16124720a3f49e212a79d44d120466bd11b5382
# 本文件是完整模块的中文阅读副本，不是独立可运行的单文件程序。
# 只展开格式；保留原算法、导入名、原路径和顶层代码。某些原模块在导入时就读取数据、运行循环或写文件，请勿直接导入。
# 实际运行需原始目录结构、对应依赖、原数据/划分/权重与CUDA环境；本次未执行训练、推理或恢复。
# 原字节与全部旧注释保存在代码核验/原始源码/历史特殊设计；此阅读版本由AST重排，普通旧注释请回原文件查看。

from pathlib import Path
import numpy as np, hashlib, json
R = Path(__file__).resolve().parent

def rng_for(kind, fold, seed, epoch, sid):
    return np.random.default_rng(int(hashlib.sha256(f'augv1|{kind}|{fold}|{seed}|{epoch}|{sid}'.encode()).hexdigest()[:16],
         16))

def selected(f, s, e, sid):
    return rng_for('switch', f, s, e, sid).random() < 0.5

class Augment:

    def __init__(self, arm, fold, seed, ids, raw, norm):
        self.arm, self.fold, self.seed, self.ids, self.raw, self.norm = (arm, fold, seed, ids, raw, norm)
        self.cache = {}

    def arrays(self, q, epoch, mask):
        target = np.array(self.raw[q], copy=True)
        inputs = target.copy()
        changed = 0
        noise_sq = 0.0
        noise_n = 0
        clipped = 0
        window_starts = []
        for k, i in enumerate(q):
            sid = str(self.ids[i])
            if self.arm not in ['G1', 'G2'] or not selected(self.fold, self.seed, epoch, sid):
                continue
            changed += 1
            if self.arm == 'G1':
                u, v = np.triu_indices(116, 1)
                r = rng_for('noise', self.fold, self.seed, epoch, sid)
                eps = r.normal(0, 0.02, len(u))
                visible = ~mask[k, 0, u, v]
                u, v, eps = (u[visible], v[visible], eps[visible])
                a = inputs[k, u, v] + eps
                clipped += int(np.sum(np.abs(a) > 1))
                a = np.clip(a, -1, 1)
                actual = a - inputs[k, u, v]
                inputs[k, u, v] = a
                inputs[k, v, u] = a
                noise_sq += float(np.sum(actual ** 2))
                noise_n += len(a)
            else:
                if sid not in self.cache:
                    self.cache[sid] = np.load(R / 'windows' / f'{sid}.npy', mmap_mode='r')
                w = self.cache[sid]
                start = int(rng_for('window', self.fold, self.seed, epoch, sid).integers(len(w)))
                target[k] = w[start]
                inputs[k] = target[k]
                window_starts.append([sid, start])
        return (inputs,
             target,
             {'augmented': changed,
             'subjects': len(q),
             'noise_squared_sum': noise_sq,
             'noise_edges': noise_n,
             'noise_clipped_edges': clipped,
             'windows': window_starts})
