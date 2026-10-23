from pathlib import Path
import json,csv,html,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'results';O.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def csvout(p,rows):
 with p.open('w') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=[];curves=[];histories={}
for j in read(R/'jobs.json'):
 d=R/'runs'/j['name'];assert (d/'completion.json').exists();h=[json.loads(l) for l in (d/'history.jsonl').read_text().splitlines()];assert len(h)==100;histories[j['name']]=h;b=min(h,key=lambda r:(r['validation_mask50']['MSE'],r['epoch']));v=b['validation_mask50'];s75=read(d/'best_validation_mask75_subjects.json');last=h[-1];rows.append({**j,'best_epoch':b['epoch'],'MSE50':v['MSE'],'correlation50':v['residual_correlation'],'amplitude50':v['residual_RMS_ratio'],'MSE75_same_best':s75['MSE'],'last_MSE50':last['validation_mask50']['MSE'],'deterioration_from_best':last['validation_mask50']['MSE']-v['MSE'],'train64_val_gap_at_best':v['MSE']-b['train_mask50']['MSE'],'lr_reductions':last.get('scheduler_state',{}).get('events',0)})
 for r in h:curves.append({'job':j['name'],'epoch':r['epoch'],'lr':r['lr_last_update'],'train64_MSE':r['train_mask50']['MSE'],'validation_MSE':r['validation_mask50']['MSE'],'gradient_preclip':r['fixed_probe_gradient']['all_preclip_l2']})
csvout(O/'upstream_all230.csv',rows);csvout(O/'upstream_epoch_curves.csv',curves);means=[]
for key in sorted(set((r['arm'],r['schedule'],r['lr']) for r in rows)):
 rs=[r for r in rows if (r['arm'],r['schedule'],r['lr'])==key];assert len(rs)==10;means.append({'arm':key[0],'schedule':key[1],'lr':key[2],**{k:float(np.mean([r[k] for r in rs])) for k in ['MSE50','correlation50','amplitude50','MSE75_same_best','deterioration_from_best','best_epoch']}})
csvout(O/'upstream_group_means.csv',means)
paired=[]
for a,b,sch in [('MS','TS','fixed'),('ML','TL','fixed'),('MS','MSO','fixed'),('MS','MSZ','fixed'),('TS','TSZ','fixed')]:
 for f in range(5):
  for s in [1,2]:
   x=next(r for r in rows if r['arm']==a and r['fold']==f and r['seed']==s and r['schedule']==sch and r['lr']==.0001);y=next(r for r in rows if r['arm']==b and r['fold']==f and r['seed']==s and r['schedule']==sch and r['lr']==.0001);paired.append({'base':a,'contrast':b,'fold':f,'seed':s,**{k+'_difference':y[k]-x[k] for k in ['MSE50','correlation50','amplitude50']}})
for a in ['MS','TS','ML','TL']:
 for sch in ['official_cosine','plateau']:
  for f in range(5):
   for s in [1,2]:
    x=next(r for r in rows if r['arm']==a and r['fold']==f and r['seed']==s and r['schedule']=='fixed' and r['lr']==.0001);y=next(r for r in rows if r['arm']==a and r['fold']==f and r['seed']==s and r['schedule']==sch);paired.append({'base':a+'_fixed','contrast':a+'_'+sch,'fold':f,'seed':s,**{k+'_difference':y[k]-x[k] for k in ['MSE50','correlation50','amplitude50']}})
csvout(O/'paired_contrasts.csv',paired)
down=[read(p) for p in sorted((R/'downstream').glob('*/completion.json'))];assert len(down)==37;ds=[{'name':r['name'],**r['pooled']} for r in down];csvout(O/'downstream_oof.csv',ds)
for a in ['MS','TS','ML','TL','MSO','MSZ','TSZ']:
 fig,axs=plt.subplots(1,3,figsize=(16,4))
 for key in sorted(set((r['schedule'],r['lr']) for r in rows if r['arm']==a)):
  hs=[histories[r['name']] for r in rows if r['arm']==a and (r['schedule'],r['lr'])==key];lbl=f'{key[0]} {key[1]:g}';ys=np.array([[r['validation_mask50']['MSE'] for r in h] for h in hs]);ts=np.array([[r['train_mask50']['MSE'] for r in h] for h in hs]);line=axs[0].plot(range(1,101),ys.mean(0),label=lbl)[0];axs[0].plot(range(1,101),ts.mean(0),'--',color=line.get_color(),alpha=.6);axs[1].plot(range(1,101),np.mean([[r['lr_last_update'] for r in h] for h in hs],0),label=lbl);axs[2].plot(range(1,101),np.mean([[r['fixed_probe_gradient']['all_preclip_l2'] for r in h] for h in hs],0),label=lbl)
 for ax,title in zip(axs,['MSE: validation solid, train64 dashed','Actual learning rate (mean; individual traces in CSV)','Fixed probe preclip gradient']):ax.set_title(title);ax.set_xlabel('epoch');ax.grid(alpha=.2);ax.legend(fontsize=6)
 fig.suptitle(a+' / 5 folds x2 seeds / no smoothing');fig.tight_layout();fig.savefig(O/f'{a}_curves.png',dpi=160);plt.close(fig)
def table(rs):
 keys=list(rs[0]);return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(f'{r[k]:.6g}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in keys)+'</tr>' for r in rs)+'</table>'
body='<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif;margin:36px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}img{max-width:100%}</style><h1>逐轮验证：算法、调度、扫描与身份编码</h1><p>230次上游训练。610人开发五折双种子；66人留出未用于训练或评分。290人下游嵌套十折评价。内部最优存在选择乐观性，非独立盲测。所有记录真实采样，无平滑。</p><h2>上游</h2>'+table(means)+''.join(f'<img src="{a}_curves.png">' for a in ['MS','TS','ML','TL','MSO','MSZ','TSZ'])+'<h2>下游</h2><p>主算法组MS/TS/ML/TL按上游选择配置；若选中plateau，全610重训使用封存的逐轮中位数学习率路径，不再读取验证集。机制组MSO/MSZ/TSZ使用固定1e-4，与MSF/TSF固定1e-4参照比较。F组随机编码器与相应基础随机编码器重复，不是额外独立重复。小模型全局288维，大模型576维。</p>'+table(ds)
(O/'report.html').write_text(body);(O/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.iterdir() if p.is_file() and p.name!='manifest.json'},indent=2));print('REPORT_COMPLETE')
