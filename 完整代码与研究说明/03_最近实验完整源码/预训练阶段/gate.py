import subprocess,os,json,time
from pathlib import Path
R=Path(__file__).resolve().parent;PY='/home/wenhao/neurostorm_2d/.venv/bin/python'
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='GPU-7ac39891-108b-f897-b92f-cd51e5392bab';env['OPENBLAS_NUM_THREADS']='1'
try:
 subprocess.run([PY,'feature_tests.py'],cwd=R,env=env,check=True)
 for arm in ['MS','TS','ML','TL']:
  subprocess.run([PY,'extract.py','--arm',arm,'--seed','1','--epoch','5','--pilot'],cwd=R,env=env,check=True)
 subprocess.run([PY,'classifier.py','--protocol','protocol.json','--job','jobs/MS_s1_e5_flatten_lr.json','--preflight'],cwd=R,env=env,check=True)
 (R/'control/engineering.json').write_text(json.dumps({'status':'PASS','time':time.time(),'checks':['thermal80to65','flatten_order','610_290_disjoint','4_model_last_feature_parity','first_epoch5_frozen_extraction','classifier_real_splits_preflight']}))
 subprocess.run([PY,'supervisor.py','launch'],cwd=R,check=True)
except BaseException as e:
 (R/'control/gate_failure.json').write_text(json.dumps({'error':repr(e),'time':time.time()}));raise
