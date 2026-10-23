# 上游选参及完整610人重训
# 作用：根据开发五折双种子平均验证曲线选择共同配置和轮次，并生成重训配置。
# 输入输出：读取上游history；输出选择记录和重训任务。
# 参数与配置：MS停滞94轮、TS停滞100轮、ML固定3e-4为48轮、TL停滞97轮；停滞组重放开发学习率轨迹。
# 注意：这是当前已经完成批次的规则；不代表实时使用下游数据重新调度。
# 来源模块：schedule/select_refits.py
# 原始源码镜像：代码核验/原始源码/schedule/select_refits.py
# 源文件SHA256：ce5c79d2592c3be488e33974ee0cadce863322e96fc5c9a000ed813fca5e45d9
# 完整性：本文件保留来源模块全部可执行语句，仅格式化并增加中文注释；Python AST逐项一致。
# 运行边界：这是保留原import名称的代码阅读版本。中文文件名不能替代原模块路径；运行需原数据、划分、权重、软件环境与原工程布局。未声称独立可运行。
# 副作用提醒：含main或顶层训练/提取语句的脚本若直接执行会读写文件或使用CUDA。本次整理未执行模型。

from pathlib import Path
import json, time, statistics

R = Path(__file__).resolve().parent


def build():
    jobs = json.loads((R / 'jobs.json').read_text())
    assert len(jobs) == 230 and all(
        (R / 'runs' / j['name'] / 'completion.json').exists() for j in jobs
    )
    out = []
    chosen = []
    replays = {}
    for arm in ['MS', 'TS', 'ML', 'TL', 'MSO', 'MSZ', 'TSZ', 'MSF', 'TSF']:
        source = arm[:2] if arm.endswith('F') else arm
        choices = (
            [(0.0001, 'fixed')]
            if arm not in ['MS', 'TS', 'ML', 'TL']
            else [(l, 'fixed') for l in [0.00003, 0.0001, 0.0003]]
            + [(0.0001, 'official_cosine'), (0.0001, 'plateau')]
        )
        scores = []
        cache = {}
        for lr, sch in choices:
            hs = [
                [
                    json.loads(l)
                    for l in (R / 'runs' / j['name'] / 'history.jsonl')
                    .read_text()
                    .splitlines()
                ]
                for j in jobs
                if j['arm'] == source and j['schedule'] == sch and j['lr'] == lr
            ]
            assert len(hs) == 10 and all(len(h) == 100 for h in hs)
            cache[(lr, sch)] = hs
            for ep in range(1, 101):
                scores.append(
                    {
                        'epoch': ep,
                        'lr': lr,
                        'schedule': sch,
                        'MSE': sum(h[ep - 1]['validation_mask50']['MSE'] for h in hs)
                        / 10,
                    }
                )
        best = min(scores, key=lambda x: (x['MSE'], x['epoch'], x['lr'], x['schedule']))
        chosen.append({'arm': arm, 'selected': best, 'all_scores': scores})
        sch = best['schedule']
        if sch == 'plateau':
            hs = cache[(best['lr'], sch)]
            replays[arm] = [
                statistics.median(h[ep - 1]['lr_last_update'] for h in hs)
                for ep in range(1, 101)
            ]
            sch = 'replay'
        for seed in [1, 2]:
            out.append(
                {
                    'name': f'{arm}_full610_s{seed}',
                    'arm': arm,
                    'seed': seed,
                    'fold': 5,
                    'lr': best['lr'],
                    'schedule': sch,
                    'epochs': best['epoch'],
                }
            )
    tmp = R / 'selection.json.tmp'
    tmp.write_text(
        json.dumps(
            {
                'selection': chosen,
                'refits': out,
                'replay_lrs': replays,
                'adaptive_refit_uses_frozen_median_trajectory_not_validation': True,
                'time': time.time(),
            },
            indent=2,
        )
    )
    tmp.replace(R / 'selection.json')
    return out
