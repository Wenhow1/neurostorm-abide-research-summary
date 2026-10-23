# 610人上游重建训练与评价
# 作用：连接原始数据、遮挡、模型、AdamW、验证、best/last/周期权重与梯度日志。
# 输入输出：读取[610,116,116]；batch输入[B,1,116,116]；预测同shape；对隐藏非对角连接计损失。
# 参数与配置：开发五折488/122；两个种子；100轮；batch8；AdamW(.9,.999,eps1e-8,wd.01)；裁剪1；训练遮挡50%；每轮验证。
# 注意：训练入口包含写日志/权重及CUDA操作，属于原始运行脚本，不要仅为阅读直接运行。留出66和下游290不参与上游选参。
# 来源模块：schedule/worker.py
# 原始源码镜像：代码核验/原始源码/schedule/worker.py
# 源文件SHA256：d9f7a846f929de4d23a79aa73db859c32963ca6047ec63b12d2fc22bb43380a8
# 完整性：本文件保留来源模块全部可执行语句，仅格式化并增加中文注释；Python AST逐项一致。
# 运行边界：这是保留原import名称的代码阅读版本。中文文件名不能替代原模块路径；运行需原数据、划分、权重、软件环境与原工程布局。未声称独立可运行。
# 副作用提醒：含main或顶层训练/提取语句的脚本若直接执行会读写文件或使用CUDA。本次整理未执行模型。

import pathlib, sys, os, json, hashlib, time, signal, argparse

R = pathlib.Path(__file__).resolve().parent
S = R / 'source'
sys.path[:0] = [str(S), str(S / 'adapters')]
import numpy as np, torch
import train_fold as old
from common import put, append, sha, read
from augment import Augment
from recording_policy import POLICY, is_record_epoch, best_row

P = read(R / 'protocol.json')
STOP = False


def sig(*a):
    global STOP
    STOP = True


signal.signal(signal.SIGTERM, sig)
signal.signal(signal.SIGUSR1, sig)


def hash_encoder(m):
    h = hashlib.sha256()
    for n, p in sorted(m.encoder.state_dict().items()):
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


from models import make


def masks(m, ids, epoch, namespace, ratio):
    return torch.cat(
        [
            m.sample_mask(
                1,
                device=torch.device('cpu'),
                mask_ratio=ratio,
                generator=torch.Generator().manual_seed(
                    int(
                        hashlib.sha256(
                            f'{namespace}|{sid}|{epoch}'.encode()
                        ).hexdigest()[:15],
                        16,
                    )
                ),
            )
            for sid in ids
        ]
    )


# 主训练入口：读取当前实验protocol和splits，随后训练与固定mask验证。
def run(a):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    raw = np.load(R / 'data/FC.npy', mmap_mode='r')
    ids = np.load(R / 'data/IDs.npy')
    split = read(R / 'splits.json')[a.fold]
    tr = np.array(split['inner_train'], dtype=int)
    va = np.array(split['inner_validation'], dtype=int)
    te = np.array(split['outer_test'], dtype=int)
    assert not set(tr) & set(va) and not set(tr) & set(te) and not set(va) & set(te)
    job = f'{a.arm}_f{a.fold}_s{a.seed}_lr{a.lr:.0e}_{a.schedule}'
    out = R / a.mode / (a.tag or job)
    out.mkdir(parents=True, exist_ok=True)
    norm = {
        'mean': float(raw[tr].mean()),
        'std': float(raw[tr].std()),
        'fit_subject_ids': ids[tr].tolist(),
    }
    prior = np.asarray(raw[tr]).mean(0)

    def image(q):
        return torch.from_numpy(
            ((raw[q] - norm['mean']) / norm['std']).astype('float32')
        )[:, None].cuda()

    m = make(a.fold, a.seed, a.arm)
    enc = hash_encoder(m)
    init = old.state_hash(m)
    params = sum(p.numel() for p in m.parameters())
    dec = sum(p.numel() for p in m.decoder.parameters())
    if not (out / 'initial.pt').exists():
        old.save(
            out / 'initial.pt',
            dict(model=m.state_dict(), arm=a.arm, fold=a.fold, seed=a.seed),
        )
    m = m.cuda()
    opt = torch.optim.AdamW(
        m.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = None
    if a.schedule == 'official_cosine':
        from official_lr_scheduler import CosineAnnealingWarmUpRestarts

        total = ((len(tr) + 7) // 8) * 100
        scheduler = CosineAnnealingWarmUpRestarts(
            opt,
            first_cycle_steps=int(0.3 * total),
            cycle_mult=1,
            max_lr=a.lr,
            min_lr=1e-9,
            warmup_steps=int(0.05 * total),
            gamma=1,
        )
    if a.schedule == 'plateau':
        from plateau import Plateau

        scheduler = Plateau(opt)
    cfg = dict(
        schedule=a.schedule,
        recording_policy=POLICY,
        job=job,
        arm=a.arm,
        fold=a.fold,
        seed=a.seed,
        mode=a.mode,
        epochs=a.epochs,
        mask_ratio=P['arms'][a.arm[:2]]['mask_ratio'],
        lr=a.lr,
        batch=8,
        weight_decay=0.01,
        gradient_clip=1,
        encoder_initial_sha256=enc,
        initial_parameters_sha256=init,
        parameters=params,
        decoder_parameters=dec,
        norm=norm,
        validation_ids=ids[va].tolist(),
        outer_test_accessed=False,
        downstream_accessed=False,
        upstream_holdout_accessed=False,
        protocol_sha256=sha(R / 'protocol.json'),
    )
    if (out / 'config.json').exists():
        assert read(out / 'config.json') == cfg
    else:
        put(out / 'config.json', cfg)

    @torch.no_grad()
    def evaluate(ix, ratio, views=3):
        m.eval()
        rows = []
        mh = hashlib.sha256()
        for view in range(views):
            for start in range(0, len(ix), 8):
                q = ix[start : start + 8]
                mask = masks(m, ids[q], view, 'cpac966_fullcv_fixed_eval_v1', ratio)
                mh.update(mask.numpy().tobytes())
                z = m(image(q), pixel_mask=mask.cuda())
                score = z['score_mask'][:, 0].cpu().numpy()
                pr = z['prediction'][:, 0].cpu().numpy() * norm['std'] + norm['mean']
                assert (
                    np.array_equal(score, score.transpose(0, 2, 1))
                    and not np.diagonal(score, axis1=1, axis2=2).any()
                )
                for k, i in enumerate(q):
                    s = score[k]
                    t = raw[i][s]
                    p = pr[k][s]
                    res = t - prior[s]
                    predres = p - prior[s]
                    den = np.sqrt(np.mean(res**2))
                    corr = (
                        np.corrcoef(res, predres)[0, 1]
                        if np.std(predres) > 1e-12 and np.std(res) > 1e-12
                        else float('nan')
                    )
                    rows.append(
                        dict(
                            subject_id=str(ids[i]),
                            view=view,
                            MSE=float(np.mean((p - t) ** 2)),
                            sym_MSE=float(
                                np.mean((((pr[k] + pr[k].T) / 2)[s] - t) ** 2)
                            ),
                            meanFC_MSE=float(np.mean(res**2)),
                            residual_correlation=float(corr),
                            residual_RMS_ratio=(
                                float(np.sqrt(np.mean(predres**2)) / den)
                                if den > 1e-12
                                else float('nan')
                            ),
                            score_fraction=float(s.sum() / (116 * 115)),
                        )
                    )
        return dict(
            sym_MSE=float(np.mean([r['sym_MSE'] for r in rows])),
            MSE=float(np.mean([r['MSE'] for r in rows])),
            meanFC_MSE=float(np.mean([r['meanFC_MSE'] for r in rows])),
            residual_correlation=float(
                np.mean([r['residual_correlation'] for r in rows])
            ),
            residual_RMS_ratio=float(np.mean([r['residual_RMS_ratio'] for r in rows])),
            score_fraction=float(np.mean([r['score_fraction'] for r in rows])),
            mask_sha256=mh.hexdigest(),
            subjects=len(ix),
            views=views,
            rows=rows,
        )

    history = []
    best_snapshot = None
    start = 1
    steps = 0
    elapsed_before = 0
    if (out / 'last.pt').exists():
        ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        assert ck['config'] == cfg
        m.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        if scheduler:
            scheduler.load_state_dict(ck['scheduler'])
        best_snapshot = ck.get('best_snapshot')
        history = ck['history']
        start = ck['epoch'] + 1
        steps = ck['steps']
        elapsed_before = ck['elapsed_seconds']
        torch.set_rng_state(ck['rng_cpu'])
        torch.cuda.set_rng_state_all(ck['rng_cuda'])
        del ck
        (out / 'history.jsonl').write_text(
            ''.join(json.dumps(x) + '\n' for x in history)
        )
        if (out / 'gradients.jsonl').exists():
            lines = [
                l
                for l in (out / 'gradients.jsonl').read_text().splitlines()
                if json.loads(l)['step'] <= steps
            ]
            (out / 'gradients.jsonl').write_text('\n'.join(lines) + '\n')
    if best_snapshot is not None:
        old.save(out / 'best.pt', best_snapshot)
    aug = Augment(a.arm, a.fold, a.seed, ids, raw, norm)

    def training_forward(q, epoch, mask):
        x, t, ar = aug.arrays(q, epoch, mask.cpu().numpy())
        x = torch.from_numpy(((x - norm['mean']) / norm['std']).astype('float32'))[
            :, None
        ].cuda()
        t = torch.from_numpy(((t - norm['mean']) / norm['std']).astype('float32'))[
            :, None
        ].cuda()
        z = m(x, pixel_mask=mask)
        z['target'] = t.detach()
        return z, ar

    beg = time.monotonic()
    ratio = P['arms'][a.arm[:2]]['mask_ratio']
    namespace = f'cpac966_fullcv_train_f{a.fold}_s{a.seed}'
    if a.mode == 'benchmark':
        m.train()
        begin = time.monotonic()
        times = []
        for step in range(a.steps):
            if STOP:
                break
            q = tr[(np.arange(8) + step * 8) % len(tr)]
            t = time.monotonic()
            opt.zero_grad(set_to_none=True)
            mask = masks(m, ids[q], step + 1, namespace, ratio).cuda()
            z, ar = training_forward(q, step + 1, mask)
            loss = old.loss_mae(z)
            assert torch.isfinite(loss)
            loss.backward()
            old.gradstats(m)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            torch.cuda.synchronize()
            times.append(time.monotonic() - t)
        put(
            out / 'benchmark.json',
            dict(
                completed_steps=len(times),
                warmup_steps=10,
                seconds=time.monotonic() - begin,
                steady_seconds=sum(times[10:]),
                steady_samples=max(0, len(times) - 10) * 8,
                initial_encoder_sha256=enc,
                decoder_parameters=dec,
            ),
        )
        return
    for epoch in range(start, a.epochs + 1):
        if STOP:
            return
        m.train()
        if a.schedule == 'replay':
            replay = read(R / 'selection.json')['replay_lrs'][a.arm]
            for g in opt.param_groups:
                g['lr'] = replay[epoch - 1]
        order = np.random.default_rng(
            2026102100 + a.fold * 10000 + epoch + (a.seed - 1) * 1000000
        ).permutation(tr)
        total = 0.0
        seen = 0
        ebeg = time.monotonic()
        for off in range(0, len(order), 8):
            if STOP:
                put(
                    out / 'paused.json',
                    dict(
                        epoch_rolled_back=epoch,
                        last_complete_epoch=epoch - 1,
                        reason='signal',
                        time_unix=time.time(),
                    ),
                )
                return
            q = order[off : off + 8]
            opt.zero_grad(set_to_none=True)
            mask = masks(m, ids[q], epoch, namespace, ratio).cuda()
            z, ar = training_forward(q, epoch, mask)
            loss = old.loss_mae(z)
            assert torch.isfinite(loss)
            loss.backward()
            gs = old.gradstats(m)
            assert gs['encoder_preclip_l2'] > 0 and gs['decoder_preclip_l2'] > 0
            used_lr = opt.param_groups[0]['lr']
            gn = float(
                torch.nn.utils.clip_grad_norm_(
                    m.parameters(), 1.0, error_if_nonfinite=True
                )
            )
            before = (
                [p.detach().clone() for p in m.parameters()]
                if steps % 10 == 0
                else None
            )
            opt.step()
            if scheduler and a.schedule != 'plateau':
                scheduler.step()
            steps += 1
            rec = dict(
                lr=used_lr,
                postclip_l2=min(gn, 1.0),
                epoch=epoch,
                step=steps,
                loss=float(loss),
                clipped=gn > 1,
                **gs,
            )
            if before is not None:
                rec['parameter_update_l2'] = float(
                    torch.sqrt(
                        sum(
                            (
                                (p.detach() - b).double().square().sum()
                                for p, b in zip(m.parameters(), before)
                            ),
                            torch.zeros((), device='cuda'),
                        )
                    )
                )
                del before
            rec['augmentation'] = ar
            append(out / 'gradients.jsonl', rec)
            total += float(loss) * len(q)
            seen += len(q)
        row = dict(
            lr_last_update=used_lr,
            lr_next_update=opt.param_groups[0]['lr'],
            epoch=epoch,
            online_training_loss=total / seen,
            online_train_raw_MSE=total / seen * norm['std'] ** 2,
            steps=steps,
            train_epoch_seconds=time.monotonic() - ebeg,
            elapsed_seconds=elapsed_before + time.monotonic() - beg,
        )
        if len(va):
            for label, ix in [('train', np.array(sorted(tr)[:64])), ('validation', va)]:
                ev = evaluate(ix, 0.5)
                row[label + '_mask50'] = {k: v for k, v in ev.items() if k != 'rows'}
                put(out / f'{label}_mask50_epoch_{epoch}.json', ev)
                if epoch == a.epochs:
                    put(out / f'{label}_mask50_subjects.json', ev)
            q = np.array(sorted(tr)[:8])
            m.eval()
            opt.zero_grad(set_to_none=True)
            z = m(
                image(q),
                pixel_mask=masks(
                    m, ids[q], 0, 'cpac966_fixed_gradient_probe_v1', 0.5
                ).cuda(),
            )
            old.loss_mae(z).backward()
            row['fixed_probe_gradient'] = old.gradstats(m)
            if hasattr(m.encoder, 'geo_gates'):
                row['coordinate_branch'] = {
                    'gates': m.encoder.geo_gates.detach().cpu().tolist(),
                    'gradient_l2': {
                        n: float(p.grad.norm()) if p.grad is not None else None
                        for n, p in m.encoder.named_parameters()
                        if n.startswith('geo_')
                    },
                }
            opt.zero_grad(set_to_none=True)
        if is_record_epoch(epoch, a.epochs) and len(va):
            ev = evaluate(va, 0.75)
            row['validation_mask75'] = {k: v for k, v in ev.items() if k != 'rows'}
            put(out / f'validation_mask75_epoch_{epoch}.json', ev)
            if epoch == a.epochs:
                put(out / 'validation_mask75_subjects.json', ev)
        if a.schedule == 'plateau' and len(va):
            scheduler.step(row['validation_mask50']['MSE'])
            row['scheduler_state'] = scheduler.state_dict()
            row['lr_next_update'] = opt.param_groups[0]['lr']
        history.append(row)
        append(out / 'history.jsonl', row)
        snapshot = None
        if is_record_epoch(epoch, a.epochs) or (
            len(va) and best_row(history)['epoch'] == epoch
        ):
            snapshot = dict(
                model={k: v.detach().cpu().clone() for k, v in m.state_dict().items()},
                epoch=epoch,
                norm=norm,
                config=cfg,
                validation_mask50=row.get('validation_mask50'),
                validation_mask75=row.get('validation_mask75'),
            )
        if is_record_epoch(epoch, a.epochs):
            old.save(out / f'epoch_{epoch}.pt', snapshot)
        if len(va) and best_row(history)['epoch'] == epoch:
            best_snapshot = snapshot
            old.save(out / 'best.pt', snapshot)
            put(
                out / 'best.json',
                {'epoch': epoch, 'MSE': row['validation_mask50']['MSE']},
            )
        old.save(
            out / 'last.pt',
            dict(
                model=m.state_dict(),
                optimizer=opt.state_dict(),
                scheduler=scheduler.state_dict() if scheduler else None,
                rng_cpu=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state_all(),
                best_snapshot=best_snapshot,
                epoch=epoch,
                steps=steps,
                config=cfg,
                norm=norm,
                history=history,
                elapsed_seconds=elapsed_before + time.monotonic() - beg,
            ),
        )
        put(
            out / 'status.json',
            dict(
                state='RUNNING',
                epoch=epoch,
                target=a.epochs,
                steps=steps,
                time_unix=time.time(),
            ),
        )
        if a.stop_after and epoch == a.stop_after:
            return
    if best_snapshot is not None:
        m.load_state_dict(best_snapshot['model'])
        put(out / 'best_validation_mask75_subjects.json', evaluate(va, 0.75))
    put(
        out / 'status.json',
        dict(state='COMPLETE', epoch=a.epochs, time_unix=time.time()),
    )
    put(
        out / 'completion.json',
        dict(
            status='COMPLETE',
            epochs=a.epochs,
            steps=steps,
            last_sha256=sha(out / 'last.pt'),
            time_unix=time.time(),
        ),
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--arm',
        choices=['MS', 'TS', 'ML', 'TL', 'MSO', 'MSZ', 'TSZ', 'MSF', 'TSF'],
        required=True,
    )
    p.add_argument(
        '--schedule',
        choices=['fixed', 'official_cosine', 'plateau', 'replay'],
        default='fixed',
    )
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument(
        '--mode', choices=['pilot', 'benchmark', 'runs', 'refits'], default='runs'
    )
    p.add_argument('--tag')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--stop-after', type=int, default=0)
    run(p.parse_args())
