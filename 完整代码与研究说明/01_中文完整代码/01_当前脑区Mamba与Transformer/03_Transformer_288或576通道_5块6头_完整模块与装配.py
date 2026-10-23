# 大小Transformer处理块与模型装配
# 作用：TransformerBlock在本文件完整定义；复用小/大脑区模型的输入投影、身份向量和输出接口，再替换编码器块。
# 输入输出：TS编码器[B,116,288]；TL编码器[B,116,576]；都重建为[B,1,116,116]。
# 参数与配置：TS与TL均5个非因果自注意力块、6头、前置LayerNorm、GELU、MLP比例4；dropout0。TS轻量线性头；TL投影至288后用2个Transformer重建块。
# 注意：此文件和前两个Mamba结构模块一起组成Transformer模型；不能把这里的23行装配误认为全模型只有23行。输入/身份参数先按同一规则初始化，然后替换块。
# 来源模块：schedule/base_models.py
# 原始源码镜像：代码核验/原始源码/schedule/base_models.py
# 源文件SHA256：ae5925783c16b496cb20cc2222a6d92b7b3a5665a25d6855b16f86263aa26560
# 完整性：本文件保留来源模块全部可执行语句，仅格式化并增加中文注释；Python AST逐项一致。
# 运行边界：这是保留原import名称的代码阅读版本。中文文件名不能替代原模块路径；运行需原数据、划分、权重、软件环境与原工程布局。未声称独立可运行。
# 副作用提醒：含main或顶层训练/提取语句的脚本若直接执行会读写文件或使用CUDA。本次整理未执行模型。

import torch
from torch import nn

# Use audited math attention on Blackwell; fused path failed repeatability pilot.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
import roi_small, roi_large


# 完整Transformer处理块：LayerNorm→自注意力→残差→LayerNorm→GELU前馈网络→残差。
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


# 模型装配：MS/ML保留Mamba；TS/TL替换为5个Transformer块；TL同时替换两个重建块。
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
