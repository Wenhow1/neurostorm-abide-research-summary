# 10_旧小块Mamba_真实或打乱坐标_完整坐标分支
# 归档主题：主题04
# 完整CoordinateEncoder：把ROI中心坐标转为对称连接几何特征，加到每个阶段输入。
# 输入/输出：116个3D坐标÷100；每条边7维=[中点3维,绝对差3维,欧氏距离1维]；7→36→36；patch平均后57×57×36。
# 本实验改动：P2先打乱ROI坐标对应；P1使用真实对应。逐阶段投影36→该层通道，乘可学习标量门，再与x和原pos相加；4个门从0开始。
# 参数在哪里：geo_mlp、geo_projections、geo_gates、forward()；coordinates_mm.npy/permutation.npy的对应由同实验atlas核验资料解释。
# 证据范围：原可学习网格位置向量仍保留；本分支不等于当前脑区模型的116个身份向量，不能混称。
# 源文件：04_位置编码与补边移窗/cpac966_coordinate_pe_20261022/coordinate_encoder.py
# 原始字节 SHA256：209a63994871f399f84db61fd4d2bdadb7a25849d6eacc89425bf6eec8352874
# 本文件是完整模块的中文阅读副本，不是独立可运行的单文件程序。
# 只展开格式；保留原算法、导入名、原路径和顶层代码。某些原模块在导入时就读取数据、运行循环或写文件，请勿直接导入。
# 实际运行需原始目录结构、对应依赖、原数据/划分/权重与CUDA环境；本次未执行训练、推理或恢复。
# 原字节与全部旧注释保存在代码核验/原始源码/历史特殊设计；此阅读版本由AST重排，普通旧注释请回原文件查看。

import torch, numpy as np
from torch import nn

class CoordinateEncoder(nn.Module):

    def __init__(self, base, arm, atlas):
        super().__init__()
        self.stages = base.stages
        self.pos_embeds = base.pos_embeds
        self.stage_dims = base.stage_dims
        self.stage_shapes = base.stage_shapes
        self.depths = base.depths
        xyz = torch.from_numpy(np.load(atlas / 'coordinates_mm.npy')).float() / 100.0
        if arm == 'P2':
            xyz = xyz[torch.from_numpy(np.load(atlas / 'permutation.npy'))]
        delta = xyz[:, None, :] - xyz[None, :, :]
        pair = torch.cat([(xyz[:,
             None,
             :] + xyz[None,
             :,
             :]) / 2,
             delta.abs(),
             delta.norm(dim=-1,
             keepdim=True)],
             dim=-1)
        self.register_buffer('geo_pair', pair[None])
        self.geo_mlp = nn.Sequential(nn.Linear(7, 36), nn.GELU(), nn.Linear(36, 36))
        self.geo_projections = nn.ModuleList([nn.Identity()] + [nn.Linear(36,
             d,
             bias=False) for d in self.stage_dims[1:]])
        self.geo_gates = nn.Parameter(torch.zeros(4))
        from neurostorm_2d.models.overlap_geometry import GridGroups2D
        self.geo_patch = GridGroups2D((116, 116), 4, 2, edge_align=False)

    def iter_blocks(self):
        for s in self.stages:
            yield from s.blocks

    def forward(self, x, *, use_prompts=False):
        g = self.geo_mlp(self.geo_pair)
        g = self.geo_patch.gather(g).mean(2).reshape(1, 57, 57, 36)
        for i, (p, stage) in enumerate(zip(self.pos_embeds, self.stages)):
            x = stage(x + p + self.geo_gates[i] * self.geo_projections[i](g), use_prompts=use_prompts)
            if stage.downsample is not None:
                g = stage.downsample.geometry.gather(g).mean(2).reshape(1, *stage.downsample.output_shape, 36)
        return x
