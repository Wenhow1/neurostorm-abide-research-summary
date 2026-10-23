"""
当前脑区token路线。每个ROI一行116连接，投影为288或576通道；加116个可学习ROI身份向量。
小Mamba：4个共权双向Mamba块，288通道；LayerNorm+Linear288→116重建头。
大Mamba：4个共权双向Mamba块，576通道；较强头的投影、两个双向块及输出层见U2。
Transformer：5个6头自注意力块，宽度288或576；大模型头中的块替换为Transformer块，见U3。
下游不使用重建解码器；encode_full输出每个ROI的表示。
阅读顺序U1/U2看完整模型，U3看Transformer替换位置，U4机制，U5行列遮挡，U6至U8训练。
对应报告05至07、09，后续展开分类与阶段检查沿用此骨干。

用途：模型与任务设计阅读，不是独立运行入口。
本文件按章节汇集真实来源模块的完整类/函数；没有执行模型、训练或读取数据。
各章节仍保留各自的import名称与类名，同名类属于不同版本，不能顺序装配。
Mamba底层选择性状态空间算子来自mamba_ssm库；PyTorch/scikit-learn基础算子不在此重复实现。
完整可恢复工程、源文件位置和校验证据见私有完整归档。
"""

raise RuntimeError("这是代码阅读文件，请勿直接执行；运行使用原工程。")


# ========================================================================
# U1 小Mamba完整模型
# 来源：01_小Mamba_288通道_4双向块_轻量线性重建头.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

import torch

from torch import nn

from mamba_ssm import Mamba

from neurostorm_2d.models.overlap_mae import OverlapMaskedAutoencoder2D

class BiBlock(nn.Module):
    def __init__(self, d=288):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.mamba = Mamba(d_model=d, d_state=16, d_conv=4, expand=2)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))

    def forward(self, x):
        z = self.norm1(x)
        f = self.mamba(z)
        r = self.mamba(z.flip(1)).flip(1)
        x = x + (f + r) * 0.5
        return x + self.mlp(self.norm2(x))

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.roi_identity = nn.Parameter(torch.empty(1, 116, 288))
        nn.init.trunc_normal_(self.roi_identity, std=0.02)
        self.blocks = nn.ModuleList([BiBlock() for _ in range(4)])
        self.norm = nn.LayerNorm(288)

    def forward(self, x):
        x = x + self.roi_identity
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

class ROIMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size = (116, 116)
        self.mask_unit = 4
        self.mask_ratio = 0.5
        self.mask_policy = 'paired_symmetric'
        self.value_projection = nn.Linear(116, 288)
        self.mask_projection = nn.Linear(116, 288, bias=False)
        self.encoder = Encoder()
        self.decoder = nn.Sequential(nn.LayerNorm(288), nn.Linear(288, 116))
        self.register_buffer(
            'off_diagonal',
            ~torch.eye(116, dtype=torch.bool)[None, None],
            persistent=False,
        )

    sample_mask = OverlapMaskedAutoencoder2D.sample_mask

    def embedded_input(self, image, mask):
        assert (
            image.shape[1:] == (1, 116, 116)
            and mask.shape == image.shape
            and mask.dtype == torch.bool
        )
        assert torch.isfinite(image).all() and torch.equal(mask, mask.transpose(-1, -2))
        return self.value_projection(
            image.masked_fill(mask, 0)[:, 0]
        ) + self.mask_projection(mask[:, 0].to(image.dtype))

    # 下游特征提取接口：完整FC、不遮挡；此处尚未平均池化。
    def encode_full(self, image, *, use_prompts=False):
        assert not use_prompts
        return self.encoder(
            self.embedded_input(image, torch.zeros_like(image, dtype=torch.bool))
        )[:, None]

    def forward(self, image, *, pixel_mask=None, generator=None):
        if pixel_mask is None:
            pixel_mask = self.sample_mask(
                len(image), device=image.device, generator=generator
            )
        z = self.encoder(self.embedded_input(image, pixel_mask))
        prediction = self.decoder(z)[:, None]
        score = pixel_mask & self.off_diagonal
        assert score.flatten(1).any(1).all()
        return {
            'prediction': prediction,
            'target': image.detach(),
            'pixel_mask': pixel_mask,
            'score_mask': score,
        }


# ========================================================================
# U2 大Mamba完整模型
# 来源：02_大Mamba_576通道_4双向块_较强重建头.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

import torch

from torch import nn

from mamba_ssm import Mamba

from neurostorm_2d.models.overlap_mae import OverlapMaskedAutoencoder2D

class BiBlock(nn.Module):
    def __init__(self, d=288):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.mamba = Mamba(d_model=d, d_state=16, d_conv=4, expand=2)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))

    def forward(self, x):
        z = self.norm1(x)
        f = self.mamba(z)
        r = self.mamba(z.flip(1)).flip(1)
        x = x + (f + r) * 0.5
        return x + self.mlp(self.norm2(x))

class Encoder(nn.Module):
    def __init__(self, d=576):
        super().__init__()
        self.roi_identity = nn.Parameter(torch.empty(1, 116, d))
        nn.init.trunc_normal_(self.roi_identity, std=0.02)
        self.blocks = nn.ModuleList([BiBlock(d) for _ in range(4)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.roi_identity
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

class ROIMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size = (116, 116)
        self.mask_unit = 4
        self.mask_ratio = 0.5
        self.mask_policy = 'paired_symmetric'
        self.value_projection = nn.Linear(116, 576)
        self.mask_projection = nn.Linear(116, 576, bias=False)
        self.encoder = Encoder()
        self.decoder = nn.Sequential(
            nn.Linear(576, 288),
            BiBlock(288),
            BiBlock(288),
            nn.LayerNorm(288),
            nn.Linear(288, 116),
        )
        self.register_buffer(
            'off_diagonal',
            ~torch.eye(116, dtype=torch.bool)[None, None],
            persistent=False,
        )

    sample_mask = OverlapMaskedAutoencoder2D.sample_mask

    def embedded_input(self, image, mask):
        assert (
            image.shape[1:] == (1, 116, 116)
            and mask.shape == image.shape
            and mask.dtype == torch.bool
        )
        assert torch.isfinite(image).all() and torch.equal(mask, mask.transpose(-1, -2))
        return self.value_projection(
            image.masked_fill(mask, 0)[:, 0]
        ) + self.mask_projection(mask[:, 0].to(image.dtype))

    def encode_full(self, image, *, use_prompts=False):
        assert not use_prompts
        return self.encoder(
            self.embedded_input(image, torch.zeros_like(image, dtype=torch.bool))
        )[:, None]

    def forward(self, image, *, pixel_mask=None, generator=None):
        if pixel_mask is None:
            pixel_mask = self.sample_mask(
                len(image), device=image.device, generator=generator
            )
        z = self.encoder(self.embedded_input(image, pixel_mask))
        prediction = self.decoder(z)[:, None]
        score = pixel_mask & self.off_diagonal
        assert score.flatten(1).any(1).all()
        return {
            'prediction': prediction,
            'target': image.detach(),
            'pixel_mask': pixel_mask,
            'score_mask': score,
        }


# ========================================================================
# U3 大小Transformer完整块与装配
# 来源：03_Transformer_288或576通道_5块6头_完整模块与装配.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

import torch

from torch import nn

import roi_small, roi_large

class TransformerBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, 6, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        z = self.norm1(x)
        x = x + self.attention(z, z, z, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))

def make(f, seed, arm):
    assert arm in ['MS', 'TS', 'ML', 'TL']
    large = arm.endswith('L')
    d = 576 if large else 288
    with torch.random.fork_rng():
        torch.manual_seed(2026102200 + f + 1000000 * (seed - 1))
        m = (roi_large if large else roi_small).ROIMAE()
        # Construct common modules first; preserve their exact paired initialization.
        if arm.startswith('T'):
            m.encoder.blocks = nn.ModuleList([TransformerBlock(d) for _ in range(5)])
            if large:
                m.decoder[1] = TransformerBlock(288)
                m.decoder[2] = TransformerBlock(288)
    return m


# ========================================================================
# U4 扫描顺序与身份向量机制
# 来源：04_脑区模型_固定扫描与身份向量机制对照.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

import torch, types

from base_models import make as base_make

ORDER = torch.randperm(116, generator=torch.Generator().manual_seed(2026102301))

def ordered_forward(self, x):
    x = (x + self.roi_identity)[:, self.scan_order]
    for block in self.blocks:
        x = block(x)
    return self.norm(x[:, self.inverse_order])

def make(f, seed, arm):
    assert arm in ['MS', 'TS', 'ML', 'TL', 'MSO', 'MSZ', 'TSZ', 'MSF', 'TSF']
    m = base_make(f, seed, arm[:2])
    if arm.endswith('O'):
        m.encoder.register_buffer('scan_order', ORDER.clone())
        m.encoder.register_buffer('inverse_order', torch.argsort(ORDER))
        m.encoder.forward = types.MethodType(ordered_forward, m.encoder)
    if arm.endswith('Z'):
        with torch.no_grad():
            m.encoder.roi_identity.zero_()
        m.encoder.roi_identity.requires_grad_(False)
    return m


# ========================================================================
# U5 按连接预算选择脑区行列遮挡
# 来源：08_脑区行列遮挡_按隐藏连接预算选ROI.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

import torch

def sample_rows(model, seed, ratio):
    # Recreate original block hidden-edge count, then use unbiased integer rounding
    # to bracket it with whole-ROI row/column masks. This matches expectation,
    # not the number of edges in every realized mask.
    gen = torch.Generator().manual_seed(seed)
    block = model.sample_mask(
        1, device=torch.device('cpu'), mask_ratio=ratio, generator=gen
    )
    n = 116
    off = ~torch.eye(n, dtype=torch.bool)[None, None]
    target = int((block & off).sum())
    counts = [n * (n - 1) - (n - k) * (n - k - 1) for k in range(n + 1)]
    hi = next(k for k, v in enumerate(counts) if v >= target)
    lo = max(0, hi - 1)
    prob = (
        (target - counts[lo]) / (counts[hi] - counts[lo])
        if counts[hi] != counts[lo]
        else 0
    )
    k = hi if float(torch.rand((), generator=gen)) < prob else lo
    selected = torch.zeros(n, dtype=torch.bool)
    selected[torch.randperm(n, generator=gen)[:k]] = True
    mask = (selected[:, None] | selected[None, :])[None, None]
    return mask, {
        'ROI_count': k,
        'hidden_edges_directed': counts[k],
        'block_target_directed': target,
        'expected_directed': counts[lo] * (1 - prob) + counts[hi] * prob,
        'hi_probability': prob,
    }


# ========================================================================
# U6 停滞5轮减半
# 来源：06_学习率_停滞5轮减半.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

import math

class Plateau:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.best = math.inf
        self.bad = 0
        self.cooldown = 0
        self.events = 0
        self.calls = 0

    def step(self, value):
        assert math.isfinite(value)
        self.calls += 1
        improved = value < self.best * 0.999
        if improved:
            self.best = value
            self.bad = 0
        elif self.cooldown == 0:
            self.bad += 1
        if self.cooldown > 0:
            self.cooldown -= 1
            self.bad = 0
            return
        if self.bad >= 5:
            for g in self.optimizer.param_groups:
                g['lr'] = max(g['lr'] * 0.5, 1e-6)
            self.events += 1
            self.bad = 0
            self.cooldown = 1

    def state_dict(self):
        return {
            k: getattr(self, k) for k in ['best', 'bad', 'cooldown', 'events', 'calls']
        }

    def load_state_dict(self, state):
        for k, v in state.items():
            setattr(self, k, v)


# ========================================================================
# U7 原余弦预热与重启调度
# 来源：07_学习率_余弦预热与周期重启.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

import math

import torch

from torch.optim import Optimizer

from torch.optim import lr_scheduler

from torch.optim.lr_scheduler import LambdaLR

from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.
    Based on https://huggingface.co/ implementation.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        t_total: int,
        cycles: float = 0.5,
        last_epoch: int = -1,
        restart_interval: int = -1,
    ) -> None:
        """
        Args:
            optimizer: wrapped optimizer.
            warmup_steps: number of warmup iterations.
            t_total: total number of training iterations.
            cycles: cosine cycles parameter.
            last_epoch: the index of last epoch.
        Returns:
            None
        """
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.restart_interval = restart_interval
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch
        )

    def lr_lambda(self, step):
        if self.restart_interval >= 0:
            step = step % self.restart_interval
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(
            max(1, self.restart_interval - self.warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress))
        )

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (
                                epoch / self.first_cycle_steps * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (
                        n
                    )
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


# ========================================================================
# U8 上游训练 验证 损失与梯度记录
# 来源：05_脑区模型_上游重建训练验证与梯度记录.py
# 本节完整保留所列类/函数；不同节是不同来源模块，同名类不代表连续覆盖设计。
# ========================================================================

import pathlib, sys, os, json, hashlib, time, signal, argparse

R = pathlib.Path(__file__).resolve().parent

S = R / 'source'

sys.path[:0] = [str(S), str(S / 'adapters')]

import numpy as np, torch

import train_fold as old

from common import put, append, sha, read

from augment import Augment

from recording_policy import POLICY, is_record_epoch, best_row

P = read(R / 'protocol.json')

STOP = False

def sig(*a):
    global STOP
    STOP = True

def hash_encoder(m):
    h = hashlib.sha256()
    for n, p in sorted(m.encoder.state_dict().items()):
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

from models import make

def masks(m, ids, epoch, namespace, ratio):
    return torch.cat(
        [
            m.sample_mask(
                1,
                device=torch.device('cpu'),
                mask_ratio=ratio,
                generator=torch.Generator().manual_seed(
                    int(
                        hashlib.sha256(
                            f'{namespace}|{sid}|{epoch}'.encode()
                        ).hexdigest()[:15],
                        16,
                    )
                ),
            )
            for sid in ids
        ]
    )

def run(a):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    raw = np.load(R / 'data/FC.npy', mmap_mode='r')
    ids = np.load(R / 'data/IDs.npy')
    split = read(R / 'splits.json')[a.fold]
    tr = np.array(split['inner_train'], dtype=int)
    va = np.array(split['inner_validation'], dtype=int)
    te = np.array(split['outer_test'], dtype=int)
    assert not set(tr) & set(va) and not set(tr) & set(te) and not set(va) & set(te)
    job = f'{a.arm}_f{a.fold}_s{a.seed}_lr{a.lr:.0e}_{a.schedule}'
    out = R / a.mode / (a.tag or job)
    out.mkdir(parents=True, exist_ok=True)
    norm = {
        'mean': float(raw[tr].mean()),
        'std': float(raw[tr].std()),
        'fit_subject_ids': ids[tr].tolist(),
    }
    prior = np.asarray(raw[tr]).mean(0)

    def image(q):
        return torch.from_numpy(
            ((raw[q] - norm['mean']) / norm['std']).astype('float32')
        )[:, None].cuda()

    m = make(a.fold, a.seed, a.arm)
    enc = hash_encoder(m)
    init = old.state_hash(m)
    params = sum(p.numel() for p in m.parameters())
    dec = sum(p.numel() for p in m.decoder.parameters())
    if not (out / 'initial.pt').exists():
        old.save(
            out / 'initial.pt',
            dict(model=m.state_dict(), arm=a.arm, fold=a.fold, seed=a.seed),
        )
    m = m.cuda()
    opt = torch.optim.AdamW(
        m.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = None
    if a.schedule == 'official_cosine':
        from official_lr_scheduler import CosineAnnealingWarmUpRestarts

        total = ((len(tr) + 7) // 8) * 100
        scheduler = CosineAnnealingWarmUpRestarts(
            opt,
            first_cycle_steps=int(0.3 * total),
            cycle_mult=1,
            max_lr=a.lr,
            min_lr=1e-9,
            warmup_steps=int(0.05 * total),
            gamma=1,
        )
    if a.schedule == 'plateau':
        from plateau import Plateau

        scheduler = Plateau(opt)
    cfg = dict(
        schedule=a.schedule,
        recording_policy=POLICY,
        job=job,
        arm=a.arm,
        fold=a.fold,
        seed=a.seed,
        mode=a.mode,
        epochs=a.epochs,
        mask_ratio=P['arms'][a.arm[:2]]['mask_ratio'],
        lr=a.lr,
        batch=8,
        weight_decay=0.01,
        gradient_clip=1,
        encoder_initial_sha256=enc,
        initial_parameters_sha256=init,
        parameters=params,
        decoder_parameters=dec,
        norm=norm,
        validation_ids=ids[va].tolist(),
        outer_test_accessed=False,
        downstream_accessed=False,
        upstream_holdout_accessed=False,
        protocol_sha256=sha(R / 'protocol.json'),
    )
    if (out / 'config.json').exists():
        assert read(out / 'config.json') == cfg
    else:
        put(out / 'config.json', cfg)

    @torch.no_grad()
    def evaluate(ix, ratio, views=3):
        m.eval()
        rows = []
        mh = hashlib.sha256()
        for view in range(views):
            for start in range(0, len(ix), 8):
                q = ix[start : start + 8]
                mask = masks(m, ids[q], view, 'cpac966_fullcv_fixed_eval_v1', ratio)
                mh.update(mask.numpy().tobytes())
                z = m(image(q), pixel_mask=mask.cuda())
                score = z['score_mask'][:, 0].cpu().numpy()
                pr = z['prediction'][:, 0].cpu().numpy() * norm['std'] + norm['mean']
                assert (
                    np.array_equal(score, score.transpose(0, 2, 1))
                    and not np.diagonal(score, axis1=1, axis2=2).any()
                )
                for k, i in enumerate(q):
                    s = score[k]
                    t = raw[i][s]
                    p = pr[k][s]
                    res = t - prior[s]
                    predres = p - prior[s]
                    den = np.sqrt(np.mean(res**2))
                    corr = (
                        np.corrcoef(res, predres)[0, 1]
                        if np.std(predres) > 1e-12 and np.std(res) > 1e-12
                        else float('nan')
                    )
                    rows.append(
                        dict(
                            subject_id=str(ids[i]),
                            view=view,
                            MSE=float(np.mean((p - t) ** 2)),
                            sym_MSE=float(
                                np.mean((((pr[k] + pr[k].T) / 2)[s] - t) ** 2)
                            ),
                            meanFC_MSE=float(np.mean(res**2)),
                            residual_correlation=float(corr),
                            residual_RMS_ratio=(
                                float(np.sqrt(np.mean(predres**2)) / den)
                                if den > 1e-12
                                else float('nan')
                            ),
                            score_fraction=float(s.sum() / (116 * 115)),
                        )
                    )
        return dict(
            sym_MSE=float(np.mean([r['sym_MSE'] for r in rows])),
            MSE=float(np.mean([r['MSE'] for r in rows])),
            meanFC_MSE=float(np.mean([r['meanFC_MSE'] for r in rows])),
            residual_correlation=float(
                np.mean([r['residual_correlation'] for r in rows])
            ),
            residual_RMS_ratio=float(np.mean([r['residual_RMS_ratio'] for r in rows])),
            score_fraction=float(np.mean([r['score_fraction'] for r in rows])),
            mask_sha256=mh.hexdigest(),
            subjects=len(ix),
            views=views,
            rows=rows,
        )

    history = []
    best_snapshot = None
    start = 1
    steps = 0
    elapsed_before = 0
    if (out / 'last.pt').exists():
        ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        assert ck['config'] == cfg
        m.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        if scheduler:
            scheduler.load_state_dict(ck['scheduler'])
        best_snapshot = ck.get('best_snapshot')
        history = ck['history']
        start = ck['epoch'] + 1
        steps = ck['steps']
        elapsed_before = ck['elapsed_seconds']
        torch.set_rng_state(ck['rng_cpu'])
        torch.cuda.set_rng_state_all(ck['rng_cuda'])
        del ck
        (out / 'history.jsonl').write_text(
            ''.join(json.dumps(x) + '\n' for x in history)
        )
        if (out / 'gradients.jsonl').exists():
            lines = [
                l
                for l in (out / 'gradients.jsonl').read_text().splitlines()
                if json.loads(l)['step'] <= steps
            ]
            (out / 'gradients.jsonl').write_text('\n'.join(lines) + '\n')
    if best_snapshot is not None:
        old.save(out / 'best.pt', best_snapshot)
    aug = Augment(a.arm, a.fold, a.seed, ids, raw, norm)

    def training_forward(q, epoch, mask):
        x, t, ar = aug.arrays(q, epoch, mask.cpu().numpy())
        x = torch.from_numpy(((x - norm['mean']) / norm['std']).astype('float32'))[
            :, None
        ].cuda()
        t = torch.from_numpy(((t - norm['mean']) / norm['std']).astype('float32'))[
            :, None
        ].cuda()
        z = m(x, pixel_mask=mask)
        z['target'] = t.detach()
        return z, ar

    beg = time.monotonic()
    ratio = P['arms'][a.arm[:2]]['mask_ratio']
    namespace = f'cpac966_fullcv_train_f{a.fold}_s{a.seed}'
    if a.mode == 'benchmark':
        m.train()
        begin = time.monotonic()
        times = []
        for step in range(a.steps):
            if STOP:
                break
            q = tr[(np.arange(8) + step * 8) % len(tr)]
            t = time.monotonic()
            opt.zero_grad(set_to_none=True)
            mask = masks(m, ids[q], step + 1, namespace, ratio).cuda()
            z, ar = training_forward(q, step + 1, mask)
            loss = old.loss_mae(z)
            assert torch.isfinite(loss)
            loss.backward()
            old.gradstats(m)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            torch.cuda.synchronize()
            times.append(time.monotonic() - t)
        put(
            out / 'benchmark.json',
            dict(
                completed_steps=len(times),
                warmup_steps=10,
                seconds=time.monotonic() - begin,
                steady_seconds=sum(times[10:]),
                steady_samples=max(0, len(times) - 10) * 8,
                initial_encoder_sha256=enc,
                decoder_parameters=dec,
            ),
        )
        return
    for epoch in range(start, a.epochs + 1):
        if STOP:
            return
        m.train()
        if a.schedule == 'replay':
            replay = read(R / 'selection.json')['replay_lrs'][a.arm]
            for g in opt.param_groups:
                g['lr'] = replay[epoch - 1]
        order = np.random.default_rng(
            2026102100 + a.fold * 10000 + epoch + (a.seed - 1) * 1000000
        ).permutation(tr)
        total = 0.0
        seen = 0
        ebeg = time.monotonic()
        for off in range(0, len(order), 8):
            if STOP:
                put(
                    out / 'paused.json',
                    dict(
                        epoch_rolled_back=epoch,
                        last_complete_epoch=epoch - 1,
                        reason='signal',
                        time_unix=time.time(),
                    ),
                )
                return
            q = order[off : off + 8]
            opt.zero_grad(set_to_none=True)
            mask = masks(m, ids[q], epoch, namespace, ratio).cuda()
            z, ar = training_forward(q, epoch, mask)
            loss = old.loss_mae(z)
            assert torch.isfinite(loss)
            loss.backward()
            gs = old.gradstats(m)
            assert gs['encoder_preclip_l2'] > 0 and gs['decoder_preclip_l2'] > 0
            used_lr = opt.param_groups[0]['lr']
            gn = float(
                torch.nn.utils.clip_grad_norm_(
                    m.parameters(), 1.0, error_if_nonfinite=True
                )
            )
            before = (
                [p.detach().clone() for p in m.parameters()]
                if steps % 10 == 0
                else None
            )
            opt.step()
            if scheduler and a.schedule != 'plateau':
                scheduler.step()
            steps += 1
            rec = dict(
                lr=used_lr,
                postclip_l2=min(gn, 1.0),
                epoch=epoch,
                step=steps,
                loss=float(loss),
                clipped=gn > 1,
                **gs,
            )
            if before is not None:
                rec['parameter_update_l2'] = float(
                    torch.sqrt(
                        sum(
                            (
                                (p.detach() - b).double().square().sum()
                                for p, b in zip(m.parameters(), before)
                            ),
                            torch.zeros((), device='cuda'),
                        )
                    )
                )
                del before
            rec['augmentation'] = ar
            append(out / 'gradients.jsonl', rec)
            total += float(loss) * len(q)
            seen += len(q)
        row = dict(
            lr_last_update=used_lr,
            lr_next_update=opt.param_groups[0]['lr'],
            epoch=epoch,
            online_training_loss=total / seen,
            online_train_raw_MSE=total / seen * norm['std'] ** 2,
            steps=steps,
            train_epoch_seconds=time.monotonic() - ebeg,
            elapsed_seconds=elapsed_before + time.monotonic() - beg,
        )
        if len(va):
            for label, ix in [('train', np.array(sorted(tr)[:64])), ('validation', va)]:
                ev = evaluate(ix, 0.5)
                row[label + '_mask50'] = {k: v for k, v in ev.items() if k != 'rows'}
                put(out / f'{label}_mask50_epoch_{epoch}.json', ev)
                if epoch == a.epochs:
                    put(out / f'{label}_mask50_subjects.json', ev)
            q = np.array(sorted(tr)[:8])
            m.eval()
            opt.zero_grad(set_to_none=True)
            z = m(
                image(q),
                pixel_mask=masks(
                    m, ids[q], 0, 'cpac966_fixed_gradient_probe_v1', 0.5
                ).cuda(),
            )
            old.loss_mae(z).backward()
            row['fixed_probe_gradient'] = old.gradstats(m)
            if hasattr(m.encoder, 'geo_gates'):
                row['coordinate_branch'] = {
                    'gates': m.encoder.geo_gates.detach().cpu().tolist(),
                    'gradient_l2': {
                        n: float(p.grad.norm()) if p.grad is not None else None
                        for n, p in m.encoder.named_parameters()
                        if n.startswith('geo_')
                    },
                }
            opt.zero_grad(set_to_none=True)
        if is_record_epoch(epoch, a.epochs) and len(va):
            ev = evaluate(va, 0.75)
            row['validation_mask75'] = {k: v for k, v in ev.items() if k != 'rows'}
            put(out / f'validation_mask75_epoch_{epoch}.json', ev)
            if epoch == a.epochs:
                put(out / 'validation_mask75_subjects.json', ev)
        if a.schedule == 'plateau' and len(va):
            scheduler.step(row['validation_mask50']['MSE'])
            row['scheduler_state'] = scheduler.state_dict()
            row['lr_next_update'] = opt.param_groups[0]['lr']
        history.append(row)
        append(out / 'history.jsonl', row)
        snapshot = None
        if is_record_epoch(epoch, a.epochs) or (
            len(va) and best_row(history)['epoch'] == epoch
        ):
            snapshot = dict(
                model={k: v.detach().cpu().clone() for k, v in m.state_dict().items()},
                epoch=epoch,
                norm=norm,
                config=cfg,
                validation_mask50=row.get('validation_mask50'),
                validation_mask75=row.get('validation_mask75'),
            )
        if is_record_epoch(epoch, a.epochs):
            old.save(out / f'epoch_{epoch}.pt', snapshot)
        if len(va) and best_row(history)['epoch'] == epoch:
            best_snapshot = snapshot
            old.save(out / 'best.pt', snapshot)
            put(
                out / 'best.json',
                {'epoch': epoch, 'MSE': row['validation_mask50']['MSE']},
            )
        old.save(
            out / 'last.pt',
            dict(
                model=m.state_dict(),
                optimizer=opt.state_dict(),
                scheduler=scheduler.state_dict() if scheduler else None,
                rng_cpu=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state_all(),
                best_snapshot=best_snapshot,
                epoch=epoch,
                steps=steps,
                config=cfg,
                norm=norm,
                history=history,
                elapsed_seconds=elapsed_before + time.monotonic() - beg,
            ),
        )
        put(
            out / 'status.json',
            dict(
                state='RUNNING',
                epoch=epoch,
                target=a.epochs,
                steps=steps,
                time_unix=time.time(),
            ),
        )
        if a.stop_after and epoch == a.stop_after:
            return
    if best_snapshot is not None:
        m.load_state_dict(best_snapshot['model'])
        put(out / 'best_validation_mask75_subjects.json', evaluate(va, 0.75))
    put(
        out / 'status.json',
        dict(state='COMPLETE', epoch=a.epochs, time_unix=time.time()),
    )
    put(
        out / 'completion.json',
        dict(
            status='COMPLETE',
            epochs=a.epochs,
            steps=steps,
            last_sha256=sha(out / 'last.pt'),
            time_unix=time.time(),
        ),
    )
