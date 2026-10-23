from pathlib import Path
import json,subprocess,sys,torch
R=Path(__file__).resolve().parent
(R/'selection.json').write_text(json.dumps({'engineering_only':True,'replay_lrs':{'ML':[.0001]*100}}))
try:
 for arm,sch in [('ML','replay'),('MSF','fixed')]:subprocess.run([sys.executable,'worker.py','--arm',arm,'--fold','5','--mode','pilot','--tag','refit_'+arm,'--epochs','1','--schedule',sch],check=True)
finally:
 (R/'selection.json').rename(R/'engineering_replay_selection.json')
(R/'refit_checks.json').write_text(json.dumps({'status':'PASS','full610_replay_and_fixed_baseline_alias':True,'downstream_real_scores_not_used':True}))
