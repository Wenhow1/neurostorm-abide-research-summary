# 01_旧小块Mamba_8个学习率五折与三种子确认_完整队列
# 归档主题：主题01
# 补齐初次学习率筛选的完整控制流程；本文件是队列和结果选择程序，不是网络层定义。
# 输入/输出：模型输入为每人116×116 FC；旧E1小块编码器末端8×8×288；该队列聚合五折重建误差。
# 本实验改动：种子1完成8个学习率×5折；先固定前2个学习率及各自共同轮次，再用种子2、3确认；累计60次，包含复用20次。
# 参数在哪里：同实验protocol.json、stageA.json、stageB_candidates.json；真正优化器与loss实现见同目录neurostorm_2d/scripts/train_mae.py。
# 证据范围：610人上游开发五折；66人保留集与290人下游均不用于本次学习率选择。
# 源文件：01_数据筛选与原始学习率/cpac966_mae_lr_extension40_20261021/extension.py
# 原始字节 SHA256：8df5dc8996a6f1d90a540cdefb485725bdeab63989406e236cbd864725df0425
# 本文件是完整模块的中文阅读副本，不是独立可运行的单文件程序。
# 只展开格式；保留原算法、导入名、原路径和顶层代码。某些原模块在导入时就读取数据、运行循环或写文件，请勿直接导入。
# 实际运行需原始目录结构、对应依赖、原数据/划分/权重与CUDA环境；本次未执行训练、推理或恢复。
# 原字节与全部旧注释保存在代码核验/原始源码/历史特殊设计；此阅读版本由AST重排，普通旧注释请回原文件查看。

"""Validated 40-fit continuation; the already-running 20 fits remain immutable."""
from pathlib import Path
import argparse, copy, json, os, subprocess, time
import numpy as np
from neurostorm_2d.overnight import parallel as m
from neurostorm_2d.overnight.supervisor import validate_manifest
R = m.ROOT

def plan():
    p = m.read(R / 'protocol.json')
    old = Path(p['source20_root'])
    assert p['combined_scientific_fits'] == 60 and p['max_scientific_fits'] == 40
    assert p['global_batch_size'] == 8 and p['world_size'] == 1 and (p['seeds'] == [1, 2, 3])
    prior = m.read(old / 'protocol.json')
    for k in ['data_files_sha256',
         'learning_rates',
         'selection_epochs',
         'mechanisms_sha256',
         'optimizer',
         'model',
         'parallel']:
        assert p[k] == prior[k], k
    A = validate_manifest(m.read(R / 'stageA.json'), R)
    candidates = validate_manifest({**A,
         'queue_id': 'extension_c_all',
         'jobs': m.read(R / 'stageB_candidates.json')['jobs']},
         R)
    selected = m.read(old / 'stageA_selection.json')['top3_learning_rates']
    C = [{**j,
         'id': j['id'].replace('B_',
         'C_',
         1)} for j in candidates['jobs'] if j['parameters']['lr'] not in selected]
    assert len(C) == 20
    validate_manifest({**A, 'queue_id': 'extension_C', 'jobs': C}, R)
    D = []
    for seed in [2, 3]:
        ds = validate_manifest(m.read(R / f'seed{seed}_candidates.json'), R)['jobs']
        assert len(ds) == 40
        D.extend(ds)
    for j in A['jobs'] + candidates['jobs'] + D:
        assert j['stage'] == 'mae' and j['epochs'] == 100 and (not j['reset_early_stop'])
        assert j['parameters']['lr'] in p['learning_rates']
        assert j['parameters']['seed'] in [1, 2, 3] and j['parameters']['batch_size'] == 8
    assert len({j['output_dir'] for j in D}) == 80
    return (p, old, A, candidates, C, D)

def metrics(job, origin, epochs):
    out = Path(job['output_dir'])
    d = m.read(out / 'completion.json')
    c = m.read(out / 'config.json')
    assert d['status'] == 'COMPLETE_MAE_LR_EXPLORATORY' and d['epochs'] == 100 and (d['optimizer_steps'] == 6100)
    assert d['protocol_sha256'] == c['protocol_sha256'] == m.sha(origin / 'protocol.json')
    assert d['last_checkpoint_sha256'] == m.sha(out / 'last.pt')
    assert c['global_batch'] == 8 and c['fixed_mask_views'] == 3 and (c['scheduler'] == 'none')
    return ({e: m.read(out / f'validation_epoch_{e}.json') for e in epochs}, c)

def rank_seed1(entries, epochs):
    curves = []
    ranking = []
    detail = []
    for lr in sorted({e['lr'] for e in entries}):
        subset = [x for x in entries if x['lr'] == lr]
        assert len(subset) == 5 and {x['config']['fold'] for x in subset} == set(range(5))
        for epoch in epochs:
            v = [x['values'][epoch]['raw_masked_mse'] for x in subset]
            curves.append({'lr': lr,
                 'common_epoch': epoch,
                 'mean_raw_MSE': float(np.mean(v)),
                 'fold_SD': float(np.std(v,
                 ddof=1))})
        best = min([x for x in curves if x['lr'] == lr], key=lambda x: (x['mean_raw_MSE'], x['common_epoch']))
        ranking.append(best)
        for x in subset:
            v = x['values'][best['common_epoch']]
            detail.append({'lr': lr,
                 'seed': 1,
                 'fold': x['config']['fold'],
                 'fixed_epoch': best['common_epoch'],
                 'raw_MSE': v['raw_masked_mse'],
                 'mean_FC_baseline': v['mean_FC_baseline_raw_mse'],
                 'initial_hash': x['config']['initial_parameters_sha256'],
                 'validation_mask_hash': v['validation_masks_sha256'],
                 'output_dir': x['output_dir']})
    for fold in range(5):
        x = [d for d in detail if d['fold'] == fold]
        assert len({d['initial_hash'] for d in x}) == len({d['validation_mask_hash'] for d in x}) == 1
    ranking.sort(key=lambda x: (x['mean_raw_MSE'], x['lr']))
    return (ranking, curves, detail)

def validate_gate():
    g = m.read(R / 'release_gate.json')
    assert g['status'] == 'PASS' and g['protocol_sha256'] == m.sha(R / 'protocol.json')
    for f, h in g['frozen_files'].items():
        assert m.sha(R / f) == h, f

def worker():
    p, old, A, candidates, C, D = plan()
    validate_gate()
    deadline = p['extension_rules']['total_deadline_unix']
    while not (old / 'final_selection.json').exists():
        state = m.read(old / 'night_status.json')
        if state['state'] == 'STOPPED_NEEDS_REVIEW':
            raise RuntimeError('Original20 stopped; no automatic restart')
        if time.time() >= deadline:
            raise TimeoutError('Shared night budget expired')
        if (m.CONTROL / 'STOP').exists():
            raise RuntimeError('STOP requested while waiting')
        m.put(R / 'night_status.json',
             {'state': 'WAITING_FOR_ORIGINAL20',
             'original_state': state['state'],
             'additional_fits': 40,
             'combined_fits': 60,
             'time_unix': time.time()})
        time.sleep(10)
    prior = m.read(old / 'final_selection.json')
    assert prior['status'] == 'COMPLETE_EXPLORATORY_LR_SELECTION' and prior['scientific_fits'] == 20
    oldA = m.read(old / 'stageA.json')['jobs']
    selected = m.read(old / 'stageA_selection.json')['top3_learning_rates']
    oldB = [j for j in m.read(old / 'stageB_candidates.json')['jobs'] if j['parameters']['lr'] in selected]
    assert len(oldA + oldB) == 20
    entries = []
    for j in oldA + oldB:
        v, c = metrics(j, old, p['selection_epochs'])
        entries.append({'lr': j['parameters']['lr'], 'values': v, 'config': c, 'output_dir': j['output_dir']})
    for suffix, cmd in [('cutoff',
         f'{m.PY} -u -m neurostorm_2d.overnight.parallel watchdog'),
         ('resources',
         f'{m.PY} -u resource_monitor.py')]:
        subprocess.run(['tmux',
             'new-session',
             '-d',
             '-s',
             f'cpac966-ext40-{suffix}',
             '-c',
             str(R),
             f'PYTHONDONTWRITEBYTECODE=1 {cmd} > control/{suffix}.log 2>&1'],
             check=True)
    m.put(R / 'phase_C_manifest.json', {**A, 'queue_id': 'extension_C', 'jobs': C})
    m.run_jobs(C, 'C_complete_all8fivefold', deadline)
    for j in C:
        v, c = metrics(j, R, p['selection_epochs'])
        entries.append({'lr': j['parameters']['lr'], 'values': v, 'config': c, 'output_dir': j['output_dir']})
    assert len(entries) == 40
    ranking, curves, detail = rank_seed1(entries, p['selection_epochs'])
    m.csvout(R / 'all8_fivefold_ranking.csv', ranking)
    m.csvout(R / 'all8_fivefold_curves.csv', curves)
    m.csvout(R / 'all8_fivefold_detail.csv', detail)
    top2 = ranking[:2]
    chosen = [v['lr'] for v in top2]
    fixed = {v['lr']: v['common_epoch'] for v in top2}
    m.put(R / 'top2_fixed_epoch_before_new_seeds.json',
         {'selected': top2,
         'selection_source': 'seed1_complete8x5',
         'fixed_before_seed2and3': True,
         'time_unix': time.time()})
    jobs = [j for j in D if j['parameters']['lr'] in chosen]
    assert len(jobs) == 20
    validate_manifest({**A, 'queue_id': 'extension_D', 'jobs': jobs}, R)
    m.put(R / 'phase_D_manifest.json', {**A, 'queue_id': 'extension_D', 'jobs': jobs})
    m.run_jobs(jobs, 'D_top2_two_more_seeds', deadline)
    final_detail = [v for v in detail if v['lr'] in chosen]
    for j in jobs:
        lr = j['parameters']['lr']
        v, c = metrics(j, R, [fixed[lr]])
        v = v[fixed[lr]]
        final_detail.append({'lr': lr,
             'seed': c['seed'],
             'fold': c['fold'],
             'fixed_epoch': fixed[lr],
             'raw_MSE': v['raw_masked_mse'],
             'mean_FC_baseline': v['mean_FC_baseline_raw_mse'],
             'initial_hash': c['initial_parameters_sha256'],
             'validation_mask_hash': v['validation_masks_sha256'],
             'output_dir': j['output_dir']})
    assert len(final_detail) == 30
    for fold in range(5):
        for seed in [1, 2, 3]:
            rows = [x for x in final_detail if x['seed'] == seed and x['fold'] == fold]
            assert len(rows) == 2 and len({x['initial_hash'] for x in rows}) == len({x['validation_mask_hash'] for x in rows}) == 1
        assert len({x['initial_hash'] for x in final_detail if x['fold'] == fold}) == 3
    summary = []
    for lr in chosen:
        means = [float(np.mean([x['raw_MSE'] for x in final_detail if x['lr'] == lr and x['seed'] == seed])) for seed in [1,
             2,
             3]]
        summary.append({'lr': lr,
             'fixed_epoch': fixed[lr],
             'mean_over3seeds5folds': float(np.mean(means)),
             'SD_of3_seed_means': float(np.std(means,
             ddof=1)),
             **{f'seed{s}_fivefold_mean': means[s - 1] for s in [1,
             2,
             3]}})
    summary.sort(key=lambda x: (x['mean_over3seeds5folds'], x['lr']))
    m.csvout(R / 'top2_three_seed_summary.csv', summary)
    m.csvout(R / 'top2_three_seed_detail.csv', final_detail)
    m.put(R / 'final_extension_result.json',
         {'status': 'COMPLETE_EXPLORATORY_60_FITS',
         'reused': 20,
         'new_fits': 40,
         'combined_fits': 60,
         'all8_seed1_ranking': ranking,
         'top2_three_seed_summary': summary,
         'selected': summary[0],
         'heldout_evaluated': False,
         'downstream_evaluated': False,
         'development_selection_not_independent_test': True,
         'time_unix': time.time()})
    m.put(R / 'night_status.json',
         {'state': 'COMPLETED',
         'combined_fits': 60,
         'new_fits': 40,
         'time_unix': time.time()})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['validate', 'launch', 'worker', 'status'])
    a = ap.parse_args()
    if a.action == 'validate':
        p, old, A, b, C, D = plan()
        print(json.dumps({'status': 'PASS',
             'reused20': str(old),
             'phase_C': len(C),
             'phase_D': 20,
             'combined60': 60,
             'seed_candidates_validated': len(D),
             'deadline': p['extension_rules']['total_deadline_unix']}))
    elif a.action == 'status':
        print(json.dumps(m.read(R / 'night_status.json'), indent=2))
    elif a.action == 'launch':
        plan()
        validate_gate()
        m.CONTROL.mkdir(exist_ok=True)
        assert not (m.CONTROL / 'extension_launched.json').exists()
        m.put(R / 'night_status.json',
             {'state': 'WAITING_FOR_ORIGINAL20',
             'new_fits': 40,
             'time_unix': time.time()})
        cmd = f'PYTHONDONTWRITEBYTECODE=1 {m.PY} -u extension.py worker > control/extension_worker.log 2>&1'
        subprocess.run(['tmux',
             'new-session',
             '-d',
             '-s',
             'cpac966-extension40-queue',
             '-c',
             str(R),
             cmd],
             check=True)
        m.put(m.CONTROL / 'extension_launched.json', {'started': time.time(), 'new_fits': 40, 'combined_fits': 60})
    else:
        try:
            worker()
        except BaseException as e:
            m.put(R / 'night_status.json',
                 {'state': 'STOPPED_NEEDS_REVIEW',
                 'error': repr(e),
                 'time_unix': time.time()})
            raise
        finally:
            (m.CONTROL / 'FINISHED').touch()
            (m.CONTROL / 'RESOURCE_MONITOR_DONE').touch()
if __name__ == '__main__':
    main()
