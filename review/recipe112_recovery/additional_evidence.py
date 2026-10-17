"""Bootstrap and preserved sensitivity/adversarial/MC evidence; no model calls."""
import time
START=time.process_time()
import json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;DATA=HERE/'snapshot'/'out'
out=dict(bootstrap_checks=[],sensitivity=[],adversarial={},count_repair=[],limitations=[])
for path in sorted(DATA.glob('fresh_*.json')):
 d=json.loads(path.read_text());H=d['H'];sample=3*H*(H+1)+3*H
 for zs,g in d['zoo'].items():
  rows=g['rows'][1:];p=np.array([r['truth'] for r in rows]);e=np.array([r['est']-r['truth'] for r in rows]);op=np.mean([r['ops'] for r in rows]);ix=np.random.default_rng(0).integers(0,len(rows),(1000,len(rows)));bs=np.mean(e[ix]**2,axis=1)/(np.mean(p[ix]*(1-p[ix]),axis=1)*sample/op);ci=np.percentile(bs,[2.5,97.5]);out['bootstrap_checks'].append(dict(file=path.name,zoo=int(zs),ci=ci.tolist(),stored=g['ci'],max_difference=float(np.max(abs(ci-g['ci'])))))
for path in sorted(DATA.glob('sens_H*.json')):
 d=json.loads(path.read_text());out['sensitivity'].append(dict(file=path.name,variants=len(d),rows=sum(len(v) for v in d.values()),frozen_worst=max(v['ratio'] for v in d['frozen'].values()),max_recorded_ratio=max((v['ratio'],name,zs) for name,g in d.items() for zs,v in g.items()),scope='Uses original broken Num wrapper for values and costs; summary-only variant results are unvalidated, no winner selection.'))
for name in ['adv_split','adv_climb','adv_diag']:
 path=DATA/(name+'.json')
 if not path.exists():out['adversarial'][name]=dict(available=False);continue
 d=json.loads(path.read_text());summary=dict(available=True,rows=len(d))
 if name=='adv_split':summary.update(max_abs_error=max(abs(r['worst_err']) for r in d),max_reimplementation_difference=max(abs(r['reimpl_diff']) for r in d),tested_H=sorted({r['H'] for r in d}),scope='Worst-neuron summaries of a diagnostic representation change, not fresh random-noise performance.')
 if name=='adv_climb':summary.update(max_abs_error=max(abs(r['err']) for r in d),scope='Adaptively attacked stored models; no parameter retuning or new attacks here.')
 out['adversarial'][name]=summary
mc=json.loads((DATA/'mc.json').read_text());z=[(r['mc']-r['exact'])/math.sqrt(max(r['exact']*(1-r['exact']),1e-12)/10000000) for r in mc];out['mc']=dict(rows=len(mc),declared_draws=10000000*len(mc),max_z_reproduction=max(abs(v-r['z']) for v,r in zip(z,mc)),mean_z=math.fsum(z)/len(z),rms_z=math.sqrt(math.fsum(v*v for v in z)/len(z)),max_abs_z=max(abs(v) for v in z),max_exact_difference=max(abs(r['mine']-r['exact']) for r in mc),scope='Saved rounded Monte Carlo means only; integer counts, RNG/checkpoint provenance and floating-forward differences not newly certified.')
for H in [4,5,6,8,16]:
 d=json.loads((HERE/f'counter_H{H}.json').read_text());assert d['complete'] and d['all_parity_pass'];rows=d['rows'];charge=sum(r['author_counter'] for r in rows);conv=3*H*(H+1)+3*H;mult=(sum(r['repaired_counter'] for r in rows)/d['sampler_original'])/(charge/conv);out['count_repair'].append(dict(H=H,rows=len(rows),corrected_multiplier=mult,max_parity=max(abs(r['parity_difference']) for r in rows),max_author_difference=max(abs(r['author_difference']) for r in rows),original_count=sum(r['saved_original']['auto'] for r in rows),repaired_count=sum(r['repaired_counter'] for r in rows),equality_count=sum(r['equality_operations'] for r in rows),cpu=d['cpu']))
out['limitations']=['Corrected125-case counters do not recount10720 fresh models.','Original sampler omits explicit final integer correctness comparison; unary negation and final float conversion arithmetic are also uncharged in recipe wrapper.','Analytic positive-volume tie defects remain unfixed in all recorded source implementations.','Bootstrap intervals condition on stored truth and original mean-cost convention; they do not certify generalization or full native cost.']
out['cpu']=time.process_time()-START;(HERE/'additional_evidence.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='bootstrap_checks'}));print('bootstrap groups',len(out['bootstrap_checks']),'max delta',max(r['max_difference'] for r in out['bootstrap_checks']))
