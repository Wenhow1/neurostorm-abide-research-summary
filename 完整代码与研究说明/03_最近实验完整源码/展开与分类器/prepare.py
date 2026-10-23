from pathlib import Path
import json,hashlib,shutil,time,numpy as np
R=Path(__file__).resolve().parent;S=R.parent/'cpac610_schedule_mechanism_20261023'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
assert not (R/'protocol.json').exists(),'refuse overwrite frozen protocol'
for d in ['features','jobs','control','logs','receipts','results']: (R/d).mkdir(exist_ok=True)
inputs={'labels':sha(S/'downstream_data/Y.npy'),'ids':sha(S/'downstream_data/IDs.npy'),'splits':sha(S/'downstream_splits.json')}
ids=np.load(S/'downstream_data/IDs.npy');assert len(ids)==290 and len(set(ids.tolist()))==290
p={'version':'2026-10-23-readout-heads-v1','data':{'source_root':str(S)},'feature_dir':'features','output_dir':'downstream','input_hashes':inputs,'feature_extraction':{'batch_size':2},'arms':['MS','TS','ML','TL'],'seeds':[1,2],'encoder_states':['pretrained','random'],'readouts':['gap','flatten'],'heads':['lr','svm','mlp'],'classifier':{'C':[.001,.003,.01,.03,.1,1],'MLP_alpha':[.001,.01,.1],'MLP_hidden':[32],'MLP_activation':'relu','MLP_max_epochs':200,'MLP_train_loss_patience':20,'MLP_lr':.001},'evaluation':{'primary_paired_readout':'gap vs flatten with same LR classifier and inner selection rule','classifier_comparison_primary':'outer fold AUROC equal-weight mean; SVM pooled scores auxiliary because scale may vary across folds','probabilities':'LR and MLP only; SVM uncalibrated decision scores, no BCE/Brier','nested_folds':'10x3','test_selection':False,'historically_inspected_cohort':True,'new_MAE_training':False,'upstream_holdout_access':False},'limits':{'cpu_parallel':4,'gpu_parallel':1,'gpu_index':1,'gpu_uuid':'GPU-7ac39891-108b-f897-b92f-cd51e5392bab','temperature_pause':80,'temperature_resume':65,'gpu0_used':False,'max_total_hours':24,'max_job_hours':6},'created_unix':time.time()}
put(R/'protocol.json',p)
# Preserve exact existing GAP features and immutable row ordering. Extraction independently rechecks these.
for a in p['arms']:
 for seed in p['seeds']:
  receipt=json.loads((S/'features'/f'{a}_s{seed}.json').read_text());assert receipt['ids_sha256']==inputs['ids']
  for kind in p['encoder_states']:
   name=f'{a}_s{seed}_{kind}';rec=next(x for x in receipt['records']if x['name']==name);src=S/'features'/f'{name}.npy';assert sha(src)==rec['feature_sha256'];shutil.copy2(src,R/'features'/f'{name}_gap.npy')
fc=np.load(S/'downstream_data/FC.npy');u,v=np.triu_indices(116,1);np.save(R/'features/R0.npy',fc[:,u,v].astype('float64'))
jobs=[]
for head in p['heads']:
 for readout in p['readouts']:
  for a in p['arms']:
   for seed in p['seeds']:
    for kind in p['encoder_states']:
     key=f'{a}_s{seed}_{kind}_{readout}';j={'name':key+'_'+head,'feature_key':key,'head':head,'arm':a,'seed':seed,'kind':kind,'readout':readout};jobs.append(j);put(R/'jobs'/(j['name']+'.json'),j)
for head in p['heads']:
 j={'name':'R0_'+head,'feature_key':'R0','head':head,'arm':'R0','readout':'raw','kind':'raw','seed':0};jobs.append(j);put(R/'jobs'/(j['name']+'.json'),j)
put(R/'jobs.json',jobs);put(R/'control/prepare_receipt.json',{'status':'PASS','inputs':inputs,'jobs':len(jobs),'subjects':290,'shape_raw':list(fc.shape),'GAP_byte_identical':True,'holdout_read':False,'time':time.time()})
print('PREPARE_PASS',len(jobs))
