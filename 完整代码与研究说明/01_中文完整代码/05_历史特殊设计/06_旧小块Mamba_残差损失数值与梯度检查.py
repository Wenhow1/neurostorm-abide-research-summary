# 06_旧小块Mamba_残差损失数值与梯度检查
# 归档主题：主题02
# 历史工程检查的完整代码：完美预测、缩放不变性、半幅度处罚、零预测有限梯度与幅度梯度方向。
# 输入/输出：本检查使用合成2×1×8×8张量；不是训练或科学结果。
# 本实验改动：仅确认损失公式和反向传播性质；不能据此声称验证或分类提升。
# 参数在哪里：引用同一历史目录的neurostorm_2d/scripts/train_mae.py。
# 证据范围：顶层立即执行检查并写结果；本次只做静态阅读，未运行本脚本。
# 源文件：02_小块模型解码器与重建目标/cpac966_residual_abc_20261021/check_loss.py
# 原始字节 SHA256：0dedd2cadcd5700fb0c9916fcd171874e3987504db5cef11c2967ece82ffbeba
# 本文件是完整模块的中文阅读副本，不是独立可运行的单文件程序。
# 只展开格式；保留原算法、导入名、原路径和顶层代码。某些原模块在导入时就读取数据、运行循环或写文件，请勿直接导入。
# 实际运行需原始目录结构、对应依赖、原数据/划分/权重与CUDA环境；本次未执行训练、推理或恢复。
# 原字节与全部旧注释保存在代码核验/原始源码/历史特殊设计；此阅读版本由AST重排，普通旧注释请回原文件查看。

import torch, json
from neurostorm_2d.scripts.train_mae import auxiliary_loss, loss_rows
from pathlib import Path
R = Path(__file__).resolve().parent
torch.manual_seed(7)
target = torch.randn(2, 1, 8, 8)
mask = (~torch.eye(8, dtype=torch.bool))[None, None].expand(2, 1, 8, 8)
prior = torch.zeros(1, 8, 8)

def out(p):
    return dict(prediction=p, target=target, score_mask=mask)
x = target.clone().requires_grad_()
assert float(auxiliary_loss(out(x), prior, 'B')) < 1e-06
assert float(auxiliary_loss(out(x), prior, 'C')) < 1e-08
x = (target * 0.5).requires_grad_()
b = auxiliary_loss(out(x), prior, 'B')
c = auxiliary_loss(out(x), prior, 'C')
assert float(b) < 1e-06 and abs(float(c) - 0.25) < 1e-06
x = torch.zeros_like(target, requires_grad=True)
loss = loss_rows(out(x)).mean() + 0.05 * auxiliary_loss(out(x), prior, 'B')
loss.backward()
assert torch.isfinite(x.grad).all()
x = (target * 0.5).requires_grad_()
auxiliary_loss(out(x), prior, 'C').backward()
assert float((x.grad * target).sum()) < 0
(R / 'loss_unit_checks.json').write_text(json.dumps(dict(status='PASS',
     checks=['perfect_prediction',
     'positive_scale_correlation_invariance',
     'half_amplitude_penalty',
     'finite_zero_prediction_gradient',
     'amplitude_gradient_direction']),
     indent=2))
print('PASS loss checks')
