# 停滞减半学习率规则
# 作用：独立保存当前任务使用的调度器，易于核对5轮停滞的精确定义。
# 输入输出：输入当前验证标量及optimizer；修改optimizer的lr。
# 参数与配置：相对改善阈值0.1%；累计5个无有效改善轮次后减半；冷却1轮；最低1e-6。
# 注意：没有使用外层测试成绩进行调度。
# 来源模块：schedule/plateau.py
# 原始源码镜像：代码核验/原始源码/schedule/plateau.py
# 源文件SHA256：755eb0d45ec32b5a46344ae8c0519ab7466e66feb8e1fd49ab6348e6f98d439e
# 完整性：本文件保留来源模块全部可执行语句，仅格式化并增加中文注释；Python AST逐项一致。
# 运行边界：这是保留原import名称的代码阅读版本。中文文件名不能替代原模块路径；运行需原数据、划分、权重、软件环境与原工程布局。未声称独立可运行。
# 副作用提醒：含main或顶层训练/提取语句的脚本若直接执行会读写文件或使用CUDA。本次整理未执行模型。

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
