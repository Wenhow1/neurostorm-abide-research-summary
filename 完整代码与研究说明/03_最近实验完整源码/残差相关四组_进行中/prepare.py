from pathlib import Path
import json,shutil,hashlib
R=Path(__file__).resolve().parent;S=R.parent/'cpac610_schedule_mechanism_20261023'
for f in ['models.py','base_models.py','roi_small.py','roi_large.py','augment.py','common.py','recording_policy.py','plateau.py','official_lr_scheduler.py','splits.json']:
 if (S/f).is_file():shutil.copy2(S/f,R/f)
 else:assert (S/'source'/f).is_file()
for f in ['source','data']:
 if not (R/f).exists():(R/f).symlink_to(S/f,target_is_directory=True)
p=json.loads((S/'protocol.json').read_text());p['residual_objective']={'lambda':None,'calibration':'shared lambda = 0.25 * min across MS/TS of median encoder gradient norm ratios MSE/correlation from 4 fixed training batches at initialization, fold0 seed1; no validation or downstream access','epoch_budget':100,'groups':['MS_base','MS_corr','TS_base','TS_corr'],'seeds':[1,2],'folds':list(range(5)),'evaluation':'Only original validation MSE controls plateau and epoch selection. No downstream tuning.'}
(R/'protocol.json').write_text(json.dumps(p,indent=2));(R/'control').mkdir(exist_ok=True);(R/'logs').mkdir(exist_ok=True)
print('PREPARED')
