from pathlib import Path
import subprocess,os,time,json,signal,sys
R=Path(__file__).resolve().parent;records=[];thermal=[]
for slots in [2,4]:
 pending=[(g,a) for g in [0,1] for a in ['MS','ML','TS','TL']];active=[];starts={};ends={};maximum={0:0,1:0}
 while pending or active:
  temps=list(map(float,subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).splitlines()))
  thermal.append({'time':time.time(),'slots':slots,'temps':temps})
  for g in [0,1]:maximum[g]=max(maximum[g],temps[g])
  if max(temps)>=80:
   for p,g,a,l in active:os.killpg(p.pid,signal.SIGTERM)
   raise RuntimeError('benchmark thermal80 stop')
  for p,g,a,l in list(active):
   if p.poll() is not None:
    l.close();assert p.returncode==0;active.remove((p,g,a,l));ends[g]=time.time()
  for g in [0,1]:
   while sum(ag==g for p,ag,a,l in active)<slots and any(pg==g for pg,a in pending):
    j=next((pg,a) for pg,a in pending if pg==g);pending.remove(j);a=j[1];name=f'bench{slots}_gpu{g}_{a}';log=(R/'control'/f'{name}.log').open('w');env={**os.environ,'CUDA_VISIBLE_DEVICES':str(g),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'};p=subprocess.Popen([sys.executable,'worker.py','--arm',a,'--mode','benchmark','--tag',name,'--steps','400'],env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active.append((p,g,a,log));starts.setdefault(g,time.time())
  time.sleep(1)
 for g in [0,1]:records.append({'gpu':g,'slots':slots,'total_steps':1600,'wall_seconds':ends[g]-starts[g],'max_temperature':maximum[g],'steps_per_second':1600/(ends[g]-starts[g])})
limits={}
for g in [0,1]:
 a=next(r for r in records if r['gpu']==g and r['slots']==2);b=next(r for r in records if r['gpu']==g and r['slots']==4);limits[str(g)]=4 if b['steps_per_second']>a['steps_per_second']*1.05 and b['max_temperature']<77 else 2
(R/'parallel_test.json').write_text(json.dumps({'records':records,'limits':limits,'criterion':'4 retained only if >5% totalthroughput gain and peak<77; runtime cap to2 at77,80pause65resume','thermal':thermal},indent=2));(R/'resource_policy.json').write_text(json.dumps({'parallel':limits,'temperature_pause':80,'temperature_resume':65,'downshift_at':77,'cutoff':'2026-10-23T09:00:00+08:00'},indent=2));print(json.dumps({'records':records,'limits':limits}),flush=True)
