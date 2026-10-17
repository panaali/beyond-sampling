"""Frozen256-index initial panel, only after independent control/source gate."""
import time
START=time.process_time()
import hashlib,json,math,resource
import numpy as np
import controls as Q
import kernel as K
HERE=Q.HERE;SOURCE=HERE.parent/'leaf_mass_review';E=Q.E
class Ops:
 def __init__(self):self.n=0
 def add(self,n):self.n+=n

def main():
 assert json.loads((HERE/'controls.json').read_text())['gate_pass'];E._RULES=Q.RULES
 z=np.load(SOURCE/'error_captured.npz');rays=z['rays'];assert len(rays)==35098
 indices=[j*35097//255 for j in range(256)];reference={};savedsource={}
 for b in range(8):
  a=np.load(SOURCE/f'error_batch_{b}.npz');reference.update(zip(a['indices'].tolist(),a['positive'].tolist()))
  a=np.load(HERE.parent/'plackett_rule_doubling'/f'batch_{b}.npz');savedsource.update(zip(a['indices'].tolist(),a['values'].tolist()))
 out=dict(indices=indices,rows=[],complete=False,hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'kernel.py',HERE/'panel.py',SOURCE/'estT_snapshot.py',SOURCE/'error_captured.npz')})
 def save():
  out['cpu']=time.process_time()-START;out['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;(HERE/'panel.json').write_text(json.dumps(out,indent=2)+'\n')
 for idx in indices:
  row=dict(index=idx,positive=reference[idx]);out['rows'].append(row);R=rays[idx];ops=Ops()
  try:
   t=time.process_time();old=E.orthant(R,ops);row.update(source=float(old),source_cpu=time.process_time()-t,source_named_mass=ops.n);assert old==savedsource[idx]
   t=time.process_time();candidate,diag=Q.value(R);row.update(candidate=float(candidate),candidate_cpu=time.process_time()-t,diagnostics=diag,difference=float(candidate-old),candidate_minus_positive=float(candidate-reference[idx]),source_minus_positive=float(old-reference[idx]))
   # Diagnostic source-inverse comparison is separately charged, not production work.
   if abs(np.linalg.det(R))>=1e-12:
    t0=time.process_time();C=Q.covariance(R);coeff,setup=K.coefficients(C);key='thin' if max(abs(C[i,j]) for i,j,_,_ in K.PAIRS)>.95 else 'wide';diff=0.
    for t,w,jac in Q.RULES[key]:
     pcs=K.correlations(coeff,float(t));O=np.linalg.inv(np.eye(4)+t*(C-np.eye(4)))
     diff=max(diff,max(abs(pc+O[k,l]/math.sqrt(O[k,k]*O[l,l])) for pc,(_,_,k,l) in zip(pcs,K.PAIRS)))
    row.update(diagnostic_max_pc_difference=diff,diagnostic_cpu=time.process_time()-t0,rule=key)
  except Exception as exc:row['failure']=repr(exc)
  if time.process_time()-START>20:out['budget_stop']=True;break
  if len(out['rows'])%32==0:save()
 out['complete']=len(out['rows'])==256;valid=[r for r in out['rows'] if 'failure' not in r];out['failures']=len(out['rows'])-len(valid)
 out['summary']=dict(candidate_source_signed=math.fsum(r['difference'] for r in valid),candidate_source_L1=math.fsum(abs(r['difference']) for r in valid),candidate_source_max=max((abs(r['difference']) for r in valid),default=None),source_positive_L1=math.fsum(abs(r['source_minus_positive']) for r in valid),candidate_positive_L1=math.fsum(abs(r['candidate_minus_positive']) for r in valid),candidate_cpu=math.fsum(r['candidate_cpu'] for r in valid),source_cpu=math.fsum(r['source_cpu'] for r in valid),setup_cpu=math.fsum(r['diagnostics'].get('setup_cpu',0) for r in valid),excursions=sum(r['diagnostics'].get('excursions',0) for r in valid),candidate_negative_count=sum(r['candidate']<0 for r in valid),source_negative_count=sum(r['source']<0 for r in valid))
 out['parity_gate_pass']=out['complete'] and out['failures']==0 and out['summary']['candidate_source_max']<=1e-10;save();print(json.dumps({k:v for k,v in out.items() if k not in ('rows','indices','hashes')}))
if __name__=='__main__':main()
