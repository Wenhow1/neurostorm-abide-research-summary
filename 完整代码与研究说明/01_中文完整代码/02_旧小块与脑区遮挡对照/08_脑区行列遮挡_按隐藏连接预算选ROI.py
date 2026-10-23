# 脑区整行整列同步遮挡
# 作用：随机选择ROI并同时遮掉该ROI的行列，离散ROI数量匹配目标隐藏边比例。
# 输入输出：单次输出布尔mask[1,1,116,116]及ROI数/连接预算等统计；worker逐人拼成[B,1,116,116]；对角线最终不评分。
# 参数与配置：50%/75%是连接隐藏预算，不是遮去一半或四分之三脑区。
# 注意：
# 来源模块：roi_rows/rowmask.py
# 原始源码镜像：代码核验/原始源码/roi_rows/rowmask.py
# 源文件SHA256：e1489db3162f4bb37352b914c003a70e220e14f31d9268c922c82faf040ac999
# 完整性：本文件保留来源模块全部可执行语句，仅格式化并增加中文注释；Python AST逐项一致。
# 运行边界：这是保留原import名称的代码阅读版本。中文文件名不能替代原模块路径；运行需原数据、划分、权重、软件环境与原工程布局。未声称独立可运行。
# 副作用提醒：含main或顶层训练/提取语句的脚本若直接执行会读写文件或使用CUDA。本次整理未执行模型。

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
