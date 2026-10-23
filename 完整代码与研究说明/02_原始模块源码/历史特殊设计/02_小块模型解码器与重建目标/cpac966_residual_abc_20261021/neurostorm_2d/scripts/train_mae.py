"""Isolated CPAC968 fixed-LR MAE entry; no downstream or held-out evaluation."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time, signal
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));t.replace(p)
def save(p,x):
    p=Path(p);t=p.with_suffix(".tmp");torch.save(x,t);t.replace(p)
def read_csv(p):
    with Path(p).open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def append(p,x):
    with Path(p).open("a") as f:f.write(json.dumps(x,allow_nan=False)+"\n");f.flush()
def state_sha(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def masks(model,ids,view,namespace):
    out=[]
    for sid in ids:
        seed=int(hashlib.sha256(f"{namespace}|{sid}|{view}".encode()).hexdigest()[:15],16)
        out.append(model.sample_mask(1,device=torch.device("cpu"),mask_ratio=.5,
                   generator=torch.Generator().manual_seed(seed)))
    return torch.cat(out)
def loss_rows(result):
    selected=result["score_mask"]
    if torch.diagonal(selected[:,0],dim1=-2,dim2=-1).any():raise ValueError("diagonal is scored")
    return torch.stack([(p-t).square().masked_select(m).mean() for p,t,m in
        zip(result["prediction"],result["target"],selected)])

def auxiliary_loss(result,prior,arm):
    terms=[]
    for prediction,target,mask in zip(result['prediction'],result['target'],result['score_mask']):
        a=(prediction-prior).masked_select(mask);b=(target-prior).masked_select(mask)
        a=a-a.mean();b=b-b.mean();va=a.square().mean();vb=b.square().mean()
        sa=torch.sqrt(va+1e-8);sb=torch.sqrt(vb+1e-8)
        correlation=(a*b).mean()/(sa*sb)
        terms.append(1-correlation if arm=='B' else (sa/sb-1).square())
    return torch.stack(terms).mean()

@torch.no_grad()
def evaluate(model,raw,ids,indices,norm,prior,device):
    model.eval();scores=np.zeros((len(indices),3));base=np.zeros_like(scores);hashes=[]
    for view in range(3):
        for start in range(0,len(indices),8):
            ix=indices[start:start+8];m=masks(model,ids[ix],view,"cpac968_fixed_validation_v1")
            assert torch.equal(m,m.transpose(-1,-2))
            image=torch.from_numpy(((raw[ix]-norm["mean"])/norm["std"]).astype("float32"))[:,None].to(device)
            r=model(image,pixel_mask=m.to(device))
            score=r["score_mask"].cpu().numpy()[:,0]
            values=loss_rows(r).cpu().numpy()*norm["std"]**2
            if not np.isfinite(values).all():raise FloatingPointError("nonfinite validation")
            scores[start:start+len(ix),view]=values
            for j,index in enumerate(ix):
                base[start+j,view]=np.mean((prior[score[j]]-raw[index][score[j]])**2)
                hashes.append(hashlib.sha256(m[j].numpy().tobytes()).hexdigest())
    model.train()
    return {"raw_masked_mse":float(scores.mean()),"mean_FC_baseline_raw_mse":float(base.mean()),
        "relative_MSE_to_mean_FC":float(scores.mean()/base.mean()),
        "validation_masks_sha256":hashlib.sha256("|".join(hashes).encode()).hexdigest(),
        "per_subject_raw_mse":scores.mean(1).tolist(),"per_subject_mean_FC_mse":base.mean(1).tolist()}

PAUSE_REQUESTED=False

def pause_signal(signum, frame):
    global PAUSE_REQUESTED
    PAUSE_REQUESTED=True

def main():
    import faulthandler
    faulthandler.dump_traceback_later(180, repeat=True)
    print("START worker",flush=True)
    signal.signal(signal.SIGTERM,pause_signal)
    signal.signal(signal.SIGUSR1,pause_signal)
    p=argparse.ArgumentParser()
    for n in ["project-dir","output-dir","fc-path","y-path","split-dir","normalization-json"]:
        p.add_argument("--"+n,type=Path,required=True)
    for n,default in [("epochs",100),("batch-size",4),("num-workers",0),("seed",1),("early-stop-patience",0),("embed-dim",36)]:
        p.add_argument("--"+n,type=int,default=default)
    for n,default in [("lr",1e-4),("weight-decay",.01),("mask-ratio",.5),("mlp-ratio",4.)]:
        p.add_argument("--"+n,type=float,default=default)
    p.add_argument("--depths",nargs="+",type=int);p.add_argument("--window-sizes",nargs="+",type=int)
    p.add_argument("--mask-policy");p.add_argument("--device",default="cuda")
    p.add_argument("--skip-figure",action="store_true");p.add_argument("--resume",action="store_true")
    args=p.parse_args();root=args.project_dir.resolve();out=args.output_dir.resolve()
    protocol=json.loads((root/"protocol.json").read_text())
    engineering=out.name.startswith("engineering_")
    if engineering:
        assert args.epochs==2 and out.name in ["engineering_B_f0_s1","engineering_C_f0_s1"]
    else:
        gate=json.loads((root/"release_gate.json").read_text())
        assert gate["status"]=="PASS" and gate["protocol_sha256"]==sha(root/"protocol.json")
        for f,h in gate["frozen_files"].items():
            assert sha(root/f)==h, "changed frozen execution source: "+f
    assert out.is_relative_to(root/"runs")
    assert args.fc_path.resolve()==root/"data/upstream/FC.npy"
    assert args.y_path.resolve()==root/"data/upstream/Y.npy"
    assert args.lr in protocol["learning_rates"] and args.epochs in [2,100]
    assert args.seed==1 and args.batch_size==8 and args.weight_decay==.01
    assert args.mask_ratio==.5 and args.mask_policy=="paired_symmetric"
    assert args.early_stop_patience==0 and args.embed_dim==36 and args.depths==[2,2,6,2]
    assert args.window_sizes==[4]*4 and args.mlp_ratio==4. and args.num_workers==0
    for rel,h in protocol["data_files_sha256"].items():
        # Downstream files are not present on the server and never accessed here.
        if rel.startswith("data/upstream/") or rel=="data/ROI_labels.json":
            assert sha(root/rel)==h
    fold=int(args.split_dir.parent.parent.name.removeprefix("f"))
    assert 0<=fold<5
    train=np.array([int(x["row_index"]) for x in read_csv(args.split_dir/"train.csv")])
    val=np.array([int(x["row_index"]) for x in read_csv(args.split_dir/"val.csv")])
    held=np.array([int(x["row_index"]) for x in read_csv(args.split_dir/"test.csv")])
    assert [len(train),len(val),len(held)]==[488,122,66]
    assert not set(train)&set(val) and not (set(train)|set(val))&set(held)
    assert set(train)|set(val)|set(held)==set(range(676))
    norm=json.loads(args.normalization_json.read_text())
    raw=np.load(args.fc_path,mmap_mode="r");ids=np.load(root/"data/upstream/IDs.npy")
    assert raw.shape==(676,116,116) and ids.shape==(676,)
    fit=np.asarray(raw[train]);assert np.isfinite(fit).all()
    assert np.isclose(fit.mean(),norm["mean"],rtol=1e-8,atol=1e-8)
    assert np.isclose(fit.std(),norm["std"],rtol=1e-8,atol=1e-8)
    prior=np.load(args.normalization_json.parent/"mean_FC_fit_only.npy")
    assert np.allclose(prior,fit.mean(0),rtol=1e-12,atol=1e-12)
    residual_scale=float(np.mean(np.var((fit-prior)[:,~np.eye(116,dtype=bool)].astype(np.float64),axis=1)) / norm["std"]**2)
    del fit
    print("START process_group",flush=True)
    assert int(os.environ.get("WORLD_SIZE","1"))==1
    print("DONE process_group",flush=True)
    rank=0;world=1
    torch.cuda.set_device(rank);device=torch.device("cuda",rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(1)
    sys.path.insert(0,str(root/"adapters"))
    from model_adapter import make_model
    print("START model",flush=True)
    model,info=make_model("MAE","A2",1,{"initialization_seed":2026102100+fold*100+1,
        "mechanisms_path":str(root/"model_source/code_v1/mechanisms.py"),
        "mechanisms_sha256":protocol["mechanisms_sha256"],"base_code_path":str(root)})
    assert not info["synthetic_mixer"] and info["dropout_probability"]==0
    model=model.to(device);initial=state_sha(model)
    initial_all=[None]*world;initial_all[0]=initial
    assert len(set(initial_all))==1
    print("START DDP",flush=True)
    ddp=model  # One process: DDP averaging is identity; avoid NCCL singleton hang.
    arm=out.name.removeprefix("engineering_").split("_")[0];assert arm in ["B","C"]
    prior_normalized=torch.from_numpy(((prior-norm["mean"])/norm["std"]).astype("float32"))[None].to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    config={"protocol_sha256":sha(root/"protocol.json"),"fold":fold,"seed":1,"lr":args.lr,
        "arm":arm,"aux_weight":.05,"training_residual_variance_normalized":residual_scale,"execution":"single_process_no_collectives","optimizer":"AdamW","weight_decay":.01,"global_batch":8,"fixed_mask_views":3,
        "fc_sha256":sha(args.fc_path),"split_train_sha256":sha(args.split_dir/"train.csv"),
        "split_val_sha256":sha(args.split_dir/"val.csv"),"normalization_sha256":sha(args.normalization_json),
        "initial_parameters_sha256":initial,"model_info":info,
        "training_subject_ids":ids[train].tolist(),"validation_subject_ids":ids[val].tolist(),
        "heldout_subject_count_not_evaluated":66,"downstream_subjects_not_loaded":290,
        "scheduler":"none","TF32":False,"precision":"FP32","engineering":engineering}
    start=1;step=0;best=float("inf");best_epoch=None;history=[]
    out.mkdir(parents=True,exist_ok=True)
    if args.resume:
        last=torch.load(out/"last.pt",map_location=device,weights_only=False)
        assert last["config"]==config
        model.load_state_dict(last["model"]);optim.load_state_dict(last["optimizer"])
        start=last["epoch"]+1;step=last["step"];best=last["best_val_loss"];best_epoch=last["best_epoch"]
        history=last["history"]
        if rank==0:
            (out/"history.jsonl").write_text("".join(json.dumps(x)+"\n" for x in history))
            if (out/"gradients.jsonl").exists():
                kept=[x for x in (out/"gradients.jsonl").read_text().splitlines() if json.loads(x)["step"]<=step]
                (out/"gradients.jsonl").write_text("\n".join(kept)+("\n" if kept else ""))
    elif rank==0:
        put(out/"config.json",config)
    pass # world_size=1; no collective needed
    started=time.monotonic()
    for epoch in range(start,args.epochs+1):
        model.train()
        permutation=np.random.default_rng(2026102100+fold*10000+epoch).permutation(train)
        assert len(permutation)%8==0
        total=torch.zeros(4,device=device,dtype=torch.float64)
        for offset in range(0,len(train),8):
            ix=permutation[offset:offset+8]
            image=torch.from_numpy(((raw[ix]-norm["mean"])/norm["std"]).astype("float32"))[:,None].to(device)
            mask=masks(model,ids[ix],epoch,f"cpac968_train_seed1_fold{fold}").to(device)
            optim.zero_grad(set_to_none=True)
            result=(model if step==0 else ddp)(image,pixel_mask=mask);main_loss=loss_rows(result).mean()
            aux=auxiliary_loss(result,prior_normalized,arm)
            weighted_aux=.05*residual_scale*aux;loss=main_loss+weighted_aux
            if step==0:
                params=[p for p in model.parameters() if p.requires_grad]
                g0=torch.autograd.grad(main_loss,params,retain_graph=True,allow_unused=True)
                g1=torch.autograd.grad(weighted_aux,params,retain_graph=True,allow_unused=True)
                gn=lambda gs:torch.sqrt(sum((g.detach().double().square().sum() for g in gs if g is not None),torch.zeros((),device=device)))
                ratio=float(gn(g1)/gn(g0).clamp_min(1e-12));assert np.isfinite(ratio) and ratio<.5, "auxiliary gradient dominates: stop for protocol review"
                put(out/'auxiliary_gradient_check.json',dict(status='PASS',main_gradient_norm=float(gn(g0)),weighted_aux_gradient_norm=float(gn(g1)),weighted_aux_to_main_gradient_norm=ratio,threshold=.5,main_loss=float(main_loss),weighted_aux=float(weighted_aux),training_only=True))
            if not torch.isfinite(loss):raise FloatingPointError("nonfinite training loss")
            loss.backward()
            if step==0:
                grads=[p.grad for name,p in model.named_parameters() if ".mamba." in name and p.grad is not None]
                assert grads and any(torch.count_nonzero(g).item()>0 for g in grads)
            pre=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            detailed=(step%10==0 and rank==0)
            if detailed:
                before=[v.detach().clone() for v in model.parameters()]
                groups={name:float(torch.sqrt(sum((p.grad.detach().double().square().sum() for n,p in model.named_parameters()
                    if p.grad is not None and n.startswith(prefix)),torch.zeros((),device=device))).item())
                    for name,prefix in [("encoder_postclip_l2","encoder."),("decoder_postclip_l2","decoder.")]}
            optim.step();step+=1
            total[0]+=loss.detach().double()*len(ix);total[1]+=len(ix);total[2]+=main_loss.detach().double()*len(ix);total[3]+=weighted_aux.detach().double()*len(ix)
            if rank==0:
                rec={"epoch":epoch,"step":step,"lr":args.lr,"preclip_gradient_l2":float(pre),
                     "clipped":bool(float(pre)>1),"all_finite":True}
                if detailed:
                    delta=torch.sqrt(sum(((p.detach()-v).double().square().sum() for p,v in zip(model.parameters(),before)),
                                         torch.zeros((),device=device)))
                    rec.update(groups,parameter_update_l2=float(delta))
                    del before
                append(out/"gradients.jsonl",rec)
        pass # world_size=1 total is already global
        pass # world_size=1; no collective needed
        if rank==0:
            rec={"epoch":epoch,"train_loss":float(total[0]/total[1]),"train_raw_mse":float(total[2]/total[1])*norm["std"]**2,"train_main_normalized_mse":float(total[2]/total[1]),"train_weighted_aux":float(total[3]/total[1]),
                 "lr":args.lr,"optimizer_steps":step,"epoch_subjects":int(total[1]),"elapsed_seconds":time.monotonic()-started}
            if epoch%5==0 or engineering or epoch==args.epochs:
                metrics=evaluate(model,raw,ids,val,norm,prior,device)
                rec["val_loss"]=metrics["raw_masked_mse"]
                rec["mean_FC_baseline_raw_mse"]=metrics["mean_FC_baseline_raw_mse"]
                rec["relative_MSE_to_mean_FC"]=metrics["relative_MSE_to_mean_FC"]
                put(out/f"validation_epoch_{epoch}.json",{"epoch":epoch,"subject_ids":ids[val].tolist(),**metrics})
                if metrics["raw_masked_mse"]<best:
                    best=metrics["raw_masked_mse"];best_epoch=epoch
                    save(out/"best.pt",{"model":model.state_dict(),"epoch":epoch,"config":config,"validation":metrics})
            history.append(rec);append(out/"history.jsonl",rec)
            checkpoint={"model":model.state_dict(),"optimizer":optim.state_dict(),"epoch":epoch,"step":step,
                        "config":config,"best_val_loss":best,"best_epoch":best_epoch,"history":history}
            save(out/"last.pt",checkpoint)
            if epoch in protocol["selection_epochs"]:
                save(out/f"epoch_{epoch}.pt",{"model":model.state_dict(),"epoch":epoch,"config":config})
            put(out/"status.json",{"state":"running","last_complete_epoch":epoch,"target_epochs":args.epochs,
                "best_epoch":best_epoch,"best_val_loss":None if not np.isfinite(best) else best,
                "elapsed_seconds":time.monotonic()-started,"actual_optimizer_steps":step})
            with (out/"loss.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,["epoch","train_loss","train_raw_mse","val_loss","lr"],extrasaction="ignore")
                writer.writeheader();writer.writerows(history)
        pass # world_size=1; no collective needed
        if PAUSE_REQUESTED or (out/"PAUSE").exists():
            put(out/"pause_receipt.json",{"state":"PAUSED_AT_EPOCH_BOUNDARY","epoch":epoch,"step":step,
                "time_unix":time.time(),"checkpoint_sha256":sha(out/"last.pt")})
            pass # no distributed group
            return
    if rank==0:
        status=json.loads((out/"status.json").read_text());status["state"]="completed";put(out/"status.json",status)
        put(out/"completion.json",{"status":"COMPLETE_MAE_LR_EXPLORATORY","epochs":args.epochs,"optimizer_steps":step,
            "last_checkpoint_sha256":sha(out/"last.pt"),"initial_parameters_sha256":initial,
            "final_parameters_sha256":state_sha(model),"protocol_sha256":sha(root/"protocol.json"),
            "heldout_evaluation":False,"classification_training":False})
    pass # world_size=1; no collective needed;pass # no distributed group

if __name__=="__main__":main()
