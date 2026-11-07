"""Reproduce the arithmetic scoring audit without editing the original source.

Requires NumPy. Set LITTLEGUY_ROOT if the repository moved. Writes fresh
results into a probe-reproduction subfolder beside this script. The source
hash used in the audit is recorded in input-manifest.json.
"""
import os
from pathlib import Path
import sys,json,numpy as np
root=Path(os.environ.get('LITTLEGUY_ROOT', '.'))
out=Path(__file__).resolve().parent / 'probe-reproduction'
out.mkdir(exist_ok=True)
src=Path(__file__).with_name('statement_fit.source.py').read_text().split('for L in LAYERS:\n    sc = oof')[0]
base=root/'.run/results/mind/lfm3-1028/science/formula-check-1104'
sys.argv=['statement_fit.py',str(out/'unused.txt'),str(base/'fc-rows-arith-hand.jsonl'),str(base/'fc-x.npz'),'tag=arith','at=num','layers=16,17']
ns={};exec(compile(src,'statement_fit_readonly','exec'),ns)
results={}
for mode in ['original','no_test_label_gate']:
 if mode=='no_test_label_gate':
  function=src[src.index('def oof('):src.index('def boot(')].replace('if yy[te].sum() == 0 or yy[tr].sum() < 3:','if yy[tr].sum() < 3:')
  exec(compile(function,'oof_without_test_labels','exec'),ns)
 results[mode]={}
 for L in ns['LAYERS']:
  print('Calculating',mode,'layer',L,flush=True)
  sc=ns['oof'](ns['X'][L],ns['y']);y=ns['y']
  a=ns['auroc'](sc[y],sc[~y]);c=ns['catch'](sc[y],sc[~y]);pw,pn,pp=ns['pairs'](sc,y)
  ai,ci=ns['boot'](sc,y)
  results[mode][str(L)]={'auroc':a,'catch_at_5pct_fpr':c,'pair_accuracy':pw,'pairs':pn,'pair_groups':pp,'auc_ci':ai.tolist(),'catch_ci':ci.tolist()}
  np.save(out/(mode+'_L'+str(L)+'_scores.npy'),sc)
  print(results[mode][str(L)],flush=True)
(out/'probe-code-audit.json').write_text(json.dumps(results,indent=2))

# The label-free text baseline uses the same corrected out-of-fold procedure.
depth=np.array([r.get('tokens_before',0) for r in ns['rows']],float)
values=np.array([abs(float(r.get('v') or 0)) for r in ns['rows']])
baseline=np.stack([np.log1p(depth),np.log10(values+1e-12)],1)
sb=ns['oof'](baseline,ns['y'],inner=False)
y=ns['y'];fold=ns['fold']
results['no_test_label_gate']['baseline']={'auroc':ns['auroc'](sb[y],sb[~y]),'catch_at_5pct_fpr':ns['catch'](sb[y],sb[~y])}
results['folds']=[{'fold':i,'n':int((fold==i).sum()),'wrong':int(y[fold==i].sum())} for i in range(5)]
(out/'probe-code-audit.json').write_text(json.dumps(results,indent=2))
print('Baseline',results['no_test_label_gate']['baseline'])
