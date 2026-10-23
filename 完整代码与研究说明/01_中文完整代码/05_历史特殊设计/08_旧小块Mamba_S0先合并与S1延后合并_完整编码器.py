# 08_旧小块Mamba_S0先合并与S1延后合并_完整编码器
# 归档主题：主题03
# 完整SameWidthMerge与OrderedEncoder，直接展示最后两个处理块和最后一次合并的次序。
# 输入/输出：共同第三阶段15×15×144→桥接15×15×288；S0先合并8×8×288再处理；S1先在15×15处理再合并；输出均8×8×288。
# 本实验改动：两组共享相同模块规格：144→288桥接、1152→288合并、末阶段全局窗口且无移窗；8×8位置向量插值为15×15再学习。
# 参数在哪里：OrderedEncoder(late=False)为S0，late=True为S1；worker.py与protocol.json负责训练/分组。
# 证据范围：S0/S1成对比较处理顺序；对原B还包含桥接和位置网格改变，不能把B对S1解释成只改一个因素。
# 源文件：03_数据增强与编码器结构/cpac966_downsample_s01_20261022/ordered_encoder.py
# 原始字节 SHA256：15975adec0b480b100306c80828f674cb54671d0b30be6a7f5bf377312a3e61e
# 本文件是完整模块的中文阅读副本，不是独立可运行的单文件程序。
# 只展开格式；保留原算法、导入名、原路径和顶层代码。某些原模块在导入时就读取数据、运行循环或写文件，请勿直接导入。
# 实际运行需原始目录结构、对应依赖、原数据/划分/权重与CUDA环境；本次未执行训练、推理或恢复。
# 原字节与全部旧注释保存在代码核验/原始源码/历史特殊设计；此阅读版本由AST重排，普通旧注释请回原文件查看。

import torch
from torch import nn
from torch.nn import functional as F

class SameWidthMerge(nn.Module):

    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry
        self.input_shape = geometry.input_shape
        self.output_shape = geometry.output_shape
        self.norm = nn.LayerNorm(1152)
        self.reduction = nn.Linear(1152, 288, bias=False)

    def forward(self, x):
        assert tuple(x.shape[1:]) == (15, 15, 288)
        groups = self.geometry.gather(x).flatten(start_dim=2)
        return self.reduction(self.norm(groups)).reshape(x.shape[0], 8, 8, 288)

class OrderedEncoder(nn.Module):
    """Shared fine-grid position before both routes; only block/merge order differs."""

    def __init__(self, base, late):
        super().__init__()
        self.late = late
        self.stage_dims = base.stage_dims
        self.depths = base.depths
        self.stage_shapes = base.stage_shapes
        self.stages = base.stages
        self.pos_embeds = base.pos_embeds
        geometry = self.stages[2].downsample.geometry
        self.stages[2].downsample = None
        self.bridge = nn.Linear(144, 288, bias=False)
        self.merge = SameWidthMerge(geometry)
        coarse = self.pos_embeds[3].detach().permute(0, 3, 1, 2)
        fine = F.interpolate(coarse,
             size=(15,
             15),
             mode='bilinear',
             align_corners=False).permute(0,
             2,
             3,
             1).contiguous()
        self.pos_embeds[3] = nn.Parameter(fine)
        for b in self.stages[3].blocks:
            b.window = (15, 15)
            b.shift = (0, 0)

    def iter_blocks(self):
        for s in self.stages:
            yield from s.blocks

    def forward(self, x, *, use_prompts=False):
        for i in range(3):
            x = self.stages[i](x + self.pos_embeds[i], use_prompts=use_prompts)
        x = self.bridge(x) + self.pos_embeds[3]
        if self.late:
            x = self.stages[3](x, use_prompts=use_prompts)
            x = self.merge(x)
        else:
            x = self.merge(x)
            x = self.stages[3](x, use_prompts=use_prompts)
        assert tuple(x.shape[1:]) == (8, 8, 288)
        return x
