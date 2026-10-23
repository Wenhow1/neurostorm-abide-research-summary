# 小Mamba完整网络结构
# 作用：定义BiBlock、Encoder、ROIMAE完整类；不是模型工厂摘录。
# 输入输出：输入[B,1,116,116]；编码器[B,116,288]；encode_full返回[B,1,116,288]；重建[B,1,116,116]。
# 参数与配置：288通道、4个共权双向Mamba块；d_state16、d_conv4、expand2；块内GELU；解码器LayerNorm288+Linear288→116。总参数5,000,084含重建头。
# 注意：实际Mamba算子来自外部mamba_ssm包，未在此伪造展开。模型本身没有下游分类器。
# 来源模块：schedule/roi_small.py
# 原始源码镜像：代码核验/原始源码/schedule/roi_small.py
# 源文件SHA256：81ad8ac42b72f8be23ba5ec30247ad286844da8807b57a6a240a2357e5e02ddb
# 完整性：本文件保留来源模块全部可执行语句，仅格式化并增加中文注释；Python AST逐项一致。
# 运行边界：这是保留原import名称的代码阅读版本。中文文件名不能替代原模块路径；运行需原数据、划分、权重、软件环境与原工程布局。未声称独立可运行。
# 副作用提醒：含main或顶层训练/提取语句的脚本若直接执行会读写文件或使用CUDA。本次整理未执行模型。

"""ROI sequence MAE, architecture-route comparison. No training outside worker."""

import torch
from torch import nn
from mamba_ssm import Mamba
from neurostorm_2d.models.overlap_mae import OverlapMaskedAutoencoder2D


# 一个双向处理块：正向和反向使用同一Mamba实例，反向输出恢复脑区顺序后求平均。
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


# 116个可学习脑区身份向量在这里加入；随后依次通过4个双向块。
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


# 完整重建模型：连接值投影+缺失位置投影→编码器→线性重建头。
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
