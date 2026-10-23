# 脑区模型及扫描身份对照
# 作用：在共同模型装配上建立主模型与机制分组。
# 输入输出：维度沿用MS/TS的288或ML/TL的576；扫描后恢复到原116脑区顺序。
# 参数与配置：主组MS、TS、ML、TL；MSO使用固定随机扫描；MSZ/TSZ将ROI身份向量置零并冻结；MSF/TSF为匹配下游参照。
# 注意：随机扫描顺序是预先固定的一份，不表示所有随机排列。
# 来源模块：schedule/models.py
# 原始源码镜像：代码核验/原始源码/schedule/models.py
# 源文件SHA256：eec1b21b800c658ef21bb615e0484b60a49160ab05eb9a98abb55c8170ea4ffb
# 完整性：本文件保留来源模块全部可执行语句，仅格式化并增加中文注释；Python AST逐项一致。
# 运行边界：这是保留原import名称的代码阅读版本。中文文件名不能替代原模块路径；运行需原数据、划分、权重、软件环境与原工程布局。未声称独立可运行。
# 副作用提醒：含main或顶层训练/提取语句的脚本若直接执行会读写文件或使用CUDA。本次整理未执行模型。

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
