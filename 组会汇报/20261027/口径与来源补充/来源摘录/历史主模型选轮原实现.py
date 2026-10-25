from pathlib import Path
import json,time,statistics
R=Path(__file__).resolve().parent
def build():
 jobs=json.loads((R/'jobs.json').read_text());assert len(jobs)==230 and all((R/'runs'/j['name']/'completion.json').exists() for j in jobs);out=[];chosen=[];replays={}
 for arm in ['MS','TS','ML','TL','MSO','MSZ','TSZ','MSF','TSF']:
  source=arm[:2] if arm.endswith('F') else arm
  choices=[(.0001,'fixed')] if arm not in ['MS','TS','ML','TL'] else [(l,'fixed') for l in [.00003,.0001,.0003]]+[(.0001,'official_cosine'),(.0001,'plateau')]
  scores=[];cache={}
  for lr,sch in choices:
   hs=[[json.loads(l) for l in (R/'runs'/j['name']/'history.jsonl').read_text().splitlines()] for j in jobs if j['arm']==source and j['schedule']==sch and j['lr']==lr];assert len(hs)==10 and all(len(h)==100 for h in hs);cache[(lr,sch)]=hs
   for ep in range(1,101):scores.append({'epoch':ep,'lr':lr,'schedule':sch,'MSE':sum(h[ep-1]['validation_mask50']['MSE'] for h in hs)/10})
  best=min(scores,key=lambda x:(x['MSE'],x['epoch'],x['lr'],x['schedule']));chosen.append({'arm':arm,'selected':best,'all_scores':scores});sch=best['schedule']
  if sch=='plateau':
   hs=cache[(best['lr'],sch)];replays[arm]=[statistics.median(h[ep-1]['lr_last_update'] for h in hs) for ep in range(1,101)];sch='replay'
  for seed in [1,2]:out.append({'name':f'{arm}_full610_s{seed}','arm':arm,'seed':seed,'fold':5,'lr':best['lr'],'schedule':sch,'epochs':best['epoch']})
 tmp=R/'selection.json.tmp';tmp.write_text(json.dumps({'selection':chosen,'refits':out,'replay_lrs':replays,'adaptive_refit_uses_frozen_median_trajectory_not_validation':True,'time':time.time()},indent=2));tmp.replace(R/'selection.json');return out
