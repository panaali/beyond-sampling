"""Independent raw-row arithmetic, without importing any estimator/review implementation."""
import time
START=time.process_time()
import json,math,hashlib
from fractions import Fraction as F
from pathlib import Path
HERE=Path(__file__).resolve().parent;DATA=HERE/'snapshot'/'out'
def avg(xs):
 xs=list(xs);return sum(xs,F(0))/len(xs)
def check(a,b,tag,failures,tol=2e-12):
 if abs(float(a)-float(b))>tol*max(1,abs(float(a)),abs(float(b))):failures.append(dict(field=tag,computed=float(a),stored=float(b)))
def main():
 out=dict(files=[],count=[],discrepancies=[],scope='Raw artifact arithmetic only; model/source provenance and native cost remain separate.');counts=json.loads((DATA/'count.json').read_text());mult={}
 for h,c in counts.items():
  rows=c['rows'];assert len(rows)==25;ratio=F(sum(r['auto'] for r in rows),c['sample_auto'])/F(sum(r['charge'] for r in rows),c['sample_conv']);check(ratio,c['mult'],f'count H{h}',out['discrepancies']);mult[int(h)]=float(ratio);out['count'].append(dict(H=int(h),rows=len(rows),sample_auto=c['sample_auto'],sample_convention=c['sample_conv'],aggregate_multiplier=float(ratio),max_reimplementation_difference=max(abs(r['diff']) for r in rows)))
 for path in sorted(DATA.glob('fresh_*.json')):
  d=json.loads(path.read_text());H=d['H'];sampler=3*H*(H+1)+3*H;f=dict(file=path.name,H=H,mode=d['mode'],sigma=d['sigma'],seeds=d['seeds'],n=d['n'],groups=[])
  for label,s in sorted(d['zoo'].items()):
   rows=s['rows'];assert rows[0]==s['clean'];nz=rows[1:];assert len(nz)==d['n']*len(d['seeds']);truth=[F(r['truth']) for r in nz];errors=[F(r['est'])-F(r['truth']) for r in nz];ops=[F(r['ops']) for r in nz];variance=[p*(1-p) for p in truth];mse=avg(x*x for x in errors);om=avg(ops);legacyvar=avg(variance)*sampler/om;individualvar=avg(v*sampler/o for v,o in zip(variance,ops));ratio=mse/legacyvar;per=[]
   for k,seed in enumerate(d['seeds']):
    sl=slice(k*d['n'],(k+1)*d['n']);per.append(float(avg(x*x for x in errors[sl])/(avg(variance[sl])*sampler/om)))
   checks=dict(ratio=ratio,mean_truth=avg(truth),rms=math.sqrt(float(mse)),bias=avg(errors),max_abs=max(abs(x) for x in errors),ops=om,exact_ops=avg(F(r['exact_ops']) for r in nz),max_reimpl_diff=max(abs(r['reimpl_diff']) for r in rows),cells=avg(F(r['cells']) for r in nz),resolved_frac=F(sum(r['resolved'] for r in nz),max(1,sum(r['cut'] for r in nz))))
   for key,value in checks.items():check(value,s[key],f'{path.name}/{label}/{key}',out['discrepancies'])
   for k,v in enumerate(per):check(v,s['per_seed'][k],f'{path.name}/{label}/per_seed{k}',out['discrepancies'])
   g=dict(zoo=int(label),noisy_models=len(nz),legacy_ratio=float(ratio),individual_budget_ratio=float(mse/individualvar),legacy_sampling_variance=float(legacyvar),individual_sampling_variance=float(individualvar),mean_squared_error=float(mse),mean_ops=float(om),per_noise_seed_legacy=per,max_reimpl_difference=checks['max_reimpl_diff'],recorded_seconds=s.get('seconds'),clean_error=rows[0]['est']-rows[0]['truth'])
   if H in mult:g['transferred_count_ratio']=float(ratio)*mult[H]
   for source,target in [('truth_mine','max_truth_difference'),('rescale_change','max_rescale_change')]:
    selected=[abs(r[source]-r['truth']) if source=='truth_mine' else abs(r[source]) for r in rows if source in r]
    if selected:g[target]=max(selected);check(g[target],s['max_truth_diff' if source=='truth_mine' else 'max_rescale_change'],f'{path.name}/{label}/{target}',out['discrepancies'])
   if all('regions' in r for r in nz):
    g.update(mean_regions=float(avg(F(r['regions']) for r in nz)),mean_half_pieces=float(avg(F(r['half_pieces']) for r in nz)),mean_cells=float(avg(F(r['cells']) for r in nz)),mean_cells_per_region=float(avg(F(r['cells'],r['regions']) for r in nz)),mean_cells_per_half_piece=float(avg(F(r['cells'],r['half_pieces']) for r in nz)))
    check(g['mean_regions'],s['regions'],f'{path.name}/{label}/regions',out['discrepancies']);check(g['mean_half_pieces'],s['half_pieces'],f'{path.name}/{label}/half_pieces',out['discrepancies'])
   f['groups'].append(g)
  out['files'].append(f)
  if time.process_time()-START>18:raise RuntimeError('aggregate18CPU cap')
 core=[f for f in out['files'] if f['H'] in (4,5,6,8) and f['mode']=='abs' and f['sigma']==.05 and f['seeds']==[80,81,82,83]]
 out['core_summary']=[dict(H=f['H'],groups=len(f['groups']),noisy_models=sum(g['noisy_models'] for g in f['groups']),worst_legacy=max(g['legacy_ratio'] for g in f['groups']),worst_individual_budget=max(g['individual_budget_ratio'] for g in f['groups']),worst_transferred_count=max(g['transferred_count_ratio'] for g in f['groups']),max_truth_difference=max(g.get('max_truth_difference',0) for g in f['groups']),max_reimplementation_difference=max(g['max_reimpl_difference'] for g in f['groups']),max_rescale_change=max(g.get('max_rescale_change',0) for g in f['groups']),mean_cells_per_region=sum(g['mean_cells_per_region'] for g in f['groups'])/len(f['groups'])) for f in core]
 out['total_groups']=sum(len(f['groups']) for f in out['files']);out['total_noisy_rows']=sum(g['noisy_models'] for f in out['files'] for g in f['groups']);out['recorded_group_wall_seconds']=sum(g.get('recorded_seconds') or 0 for f in out['files'] for g in f['groups']);out['cpu']=time.process_time()-START
 (HERE/'aggregate.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('files','count')}))
if __name__=='__main__':main()
