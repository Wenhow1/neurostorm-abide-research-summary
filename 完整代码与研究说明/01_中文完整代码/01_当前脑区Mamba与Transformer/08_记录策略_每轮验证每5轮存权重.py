# 评价与保存策略
# 作用：确认实际验证间隔、best记录及周期权重保存。
# 输入输出：处理epoch整数与history记录。
# 参数与配置：以代码字段和本次历史日志为准；best与last另存。
# 注意：部分历史版本字符串可能保留旧名字，实际字段及执行记录优先。
# 来源模块：schedule/recording_policy.py
# 原始源码镜像：代码核验/原始源码/schedule/recording_policy.py
# 源文件SHA256：f17fb84c092a79bc7ed748a2ec8155e1b31048126111eead4a901c7804838763
# 完整性：本文件保留来源模块全部可执行语句，仅格式化并增加中文注释；Python AST逐项一致。
# 运行边界：这是保留原import名称的代码阅读版本。中文文件名不能替代原模块路径；运行需原数据、划分、权重、软件环境与原工程布局。未声称独立可运行。
# 副作用提醒：含main或顶层训练/提取语句的脚本若直接执行会读写文件或使用CUDA。本次整理未执行模型。

"""Recording policy for NEW upstream experiments; no model or data access."""

POLICY = {
    'version': '2026-10-22-every5-v1',
    'validation_interval': 1,
    'checkpoint_interval': 5,
    'fixed_gradient_probe_interval': 1,
    'last_interval': 1,
    'best_metric': 'validation_mask50.MSE',
    'best_direction': 'min',
    'tie_break': 'earlier_epoch',
}


def is_record_epoch(epoch, total_epochs):
    return epoch % 5 == 0 or epoch == total_epochs


def best_row(history):
    candidates = [r for r in history if 'validation_mask50' in r]
    return (
        min(candidates, key=lambda r: (r['validation_mask50']['MSE'], r['epoch']))
        if candidates
        else None
    )
