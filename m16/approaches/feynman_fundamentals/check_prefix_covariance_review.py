import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time
import numpy as np
import mpmath as mp
P=Path('/workspace/work/recover-small_h2_56/review/arc_cumulants_negative/recovery_review');start=time.process_time();mp.mp.dps=60
trained=json.loads((P/'covariance_trained.json').read_text())['rows'][0];controls=json.loads((P/'covariance_controls.json').read_text())['rows']
for row in controls+[trained]:
 mu=np.array(row['mean']);cov=np.array(row['covariance']);ref=row['reference'];assert np.allclose(mu,np.array(ref['mean'],float),atol=1e-12,rtol=1e-10);assert np.allclose(cov,np.array(ref['covariance'],float),atol=1e-12,rtol=1e-10)
def m(q):q=F(q);return mp.mpf(q.numerator)/q.denominator
wi=list(map(F,trained['wi']));wh=[list(map(F,row)) for row in trained['wh']];a=[[sum(w*max(sign*x,0) for w,x in zip(row,wi)) for row in wh] for sign in (1,-1)]
# Independent direct positive quadrature, not author's antiderivative or all-sector matrix sum.
checks=[]
for i,j in [(0,0),(0,10),(10,13),(1,9),(6,15),(13,13)]:
 raw=mp.mpf(0)
 for aa in a:
  ai,aj,bi,bj=map(m,[aa[i],aa[j],wi[i],wi[j]]);cuts=[-mp.pi/2,mp.pi/2]
  for x,y in [(ai,bi),(aj,bj)]:
   for k in range(-2,3):
    t=mp.atan2(y,x)+mp.pi/2+k*mp.pi
    if -mp.pi/2<t<mp.pi/2:cuts.append(t)
  cuts=sorted(set(cuts))
  for l,r in zip(cuts,cuts[1:]):
   raw+=mp.quad(lambda t:max(ai*mp.cos(t)+bi*mp.sin(t),0)*max(aj*mp.cos(t)+bj*mp.sin(t),0),[l,r])/mp.pi
 mui=sum(mp.sqrt(m(aa[i])**2+m(wi[i])**2)+m(aa[i]) for aa in a)/(2*mp.sqrt(2*mp.pi));muj=sum(mp.sqrt(m(aa[j])**2+m(wi[j])**2)+m(aa[j]) for aa in a)/(2*mp.sqrt(2*mp.pi))
 cov=raw-mui*muj;ref=mp.mpf(trained['reference']['covariance'][i][j]);assert abs(cov-ref)<mp.mpf('1e-44');checks.append(dict(i=i,j=j,absolute_difference=str(abs(cov-ref))))
res=json.loads((P/'closure_result.json').read_text());assert hashlib.sha256((P/'closure_reference.npz').read_bytes()).hexdigest()==res['reference_sha256'];archive=np.load(P/'closure_reference.npz');bm=archive['block_means'];br=archive['block_raw_second'];assert bm.shape==(8,4,16)
bc=br-bm[:,:,:,None]*bm[:,:,None,:];mu=bm.mean(0);cov=br.mean(0)-mu[:,:,None]*mu[:,None,:];assert np.array_equal(mu,archive['mean']);assert np.array_equal(cov,archive['covariance'])
def metric(m,c,rm,rc):
 err=c-rc;off=~np.eye(16,dtype=bool)
 norm=lambda a:np.sqrt(np.mean(a*a))
 return dict(mean=float(norm(m-rm)/norm(rm)),variance=float(norm(np.diag(err))/norm(np.diag(rc))),off_diagonal=float(norm(err[off])/norm(rc[off])),covariance=float(norm(err)/norm(rc)))
for t,row in enumerate(res['rows']):
 names=list(row['methods']);errs=[]
 for name,r in row['methods'].items():
  mm=np.array(r['mean']);cc=np.array(r['covariance']);calc=metric(mm,cc,mu[t],cov[t]);assert calc==r['relative_rms_errors'];block=[metric(mm,cc,bm[b,t],bc[b,t]) for b in range(8)];assert block==r['block_errors'];errs.append(block)
 for key,saved in row['paired_improvements'].items():
  d=np.array([errs[0][b][key]-errs[1][b][key] for b in range(8)]);assert float(d.mean())==saved['block_mean_error_reduction'];assert float(d.std(ddof=1)/np.sqrt(8))==saved['block_standard_error']
# Reset uses checked exact-prefix array and leaves first step unchanged.
assert res['rows'][0]['methods'][names[0]]['mean']==res['rows'][0]['methods'][names[1]]['mean']
reset=res['rows'][1]['methods'][names[1]];assert reset['mean']==trained['mean'] and reset['covariance']==trained['covariance']
out=dict(trained_saved_pair_entries=256,analytic_control_models=len(controls),independent_positive_pair_quadratures=checks,reference_blocks=8,reference_samples=262144,step4_errors={n:r['relative_rms_errors'] for n,r in res['rows'][3]['methods'].items()},cpu_seconds=time.process_time()-start,scope='Source factors/series and saved aggregation; six independent MP positive pair integrals. No new model inputs, continuation rerun, confidence certificate, or final accuracy claim.')
Path(__file__).with_name('prefix_covariance_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
