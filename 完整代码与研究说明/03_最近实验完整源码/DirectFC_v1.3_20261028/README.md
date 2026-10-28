# DirectFC v1.3 实际运行代码

237项拟合已完成，193人测试尚未开放。

- models.py：Mamba1/Transformer/基础GCN，两层、64/128通道；C组头前Dropout，D组身份传播由worker传入。
- worker.py、scheduler.py：真实训练/验证、梯度/更新、Plateau、恢复。
- common.py：图、缩放、指标与选择。
- supervisor.py：双卡各最多4路、完整阶段接续，结束于FROZEN_TEST_LOCKED。
- finalize.py：开发复算、选轮、7模型冻结。
- gate_v13.py：实际v1.3验收入口；gate.py仅复用比较/参考检查函数，其历史main不是本轮入口。
- prepare_v13.py：继承已审计划分，数据未在公开仓库提供。

Mamba使用官方Mamba1模块，DIRECTFC_SCAN_REF=1选择官方selective_scan_ref在CUDA执行。该路径修复了原生扫描恢复输出一致性问题，真实前向/梯度等价及10个恢复案例通过；不是MLP替代Mamba。源码与运行资料属于本地协议适配，不宣称论文成绩复现。

代码是实际来源快照，不是含数据的即用训练包。原始FC、ID、标签映射、预测、权重和原日志不公开；不应直接执行prepare重新建队列。
