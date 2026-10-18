"""Only four preregistered analytic controls, no trained data loaded."""
import hashlib,json,time,traceback
from pathlib import Path
from fractions import Fraction as F
from core import Arithmetic,reconstruct,analyse,classify,pack
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 start=time.process_time();dst=HERE/'controls.json';assert not dst.exists();gate=json.loads((HERE/'controls_gate.json').read_text());assert gate['complete']
 for p,h in gate['pins'].items():assert sha(Path(p))==h
 out=dict(complete=False,gate_sha256=sha(HERE/'controls_gate.json'),rows=[],new_trained_forward_calls=0,new_reference_draws=0)
 def save():out['cpu']=time.process_time()-start;t=dst.with_suffix('.tmp');t.write_text(json.dumps(pack(out),indent=2)+'\n');t.replace(dst)
 save()
 try:
  for name,wo,expected in [('crossing',[[2,-1],[0,0],[0,-1]],[[-2,0,1],[0,0,0],[-4,-2,-1]]),('invariant',[[1,0],[0,0],[0,-1]],[[1,1,1],[0,0,0],[-4,-2,-1]])]:
   m=dict(H=2,T=3,D=1,wi=[1,1],wh=[[1,0],[0,2]],wo=wo);x=[1,2,3];history=[[True,True]]*3;A=Arithmetic();r=reconstruct(m,x,history,A);result=analyse(x,r,A)
   row=dict(name=name,model=m,x=x,history=history,reconstructed=r,result=result);out['rows'].append(row);save()
   assert r['L']==expected and r['denominator']==1 and r['lower']==-1 and r['upper'] is None
   if name=='crossing':assert result['roots']==[F(1)] and result['robust_opposite_pair']==[0,1] and [p['correct'] for p in result['open_probes']]==[False,True] and result['crossing_points'][0]['prediction']==0 and not result['crossing_points'][0]['robust']
   else:assert not result['roots'] and result['robust_opposite_pair'] is None and not result['open_probes'][0]['correct']
  m=dict(H=1,T=3,D=1,wi=[0],wh=[[0]],wo=[[0],[0],[0]]);x=[2,1,3];history=[[False]]*3;A=Arithmetic();r=reconstruct(m,x,history,A);result=analyse(x,r,A)
  out['rows'].append(dict(name='zero_permanent_tie',model=m,x=x,history=history,reconstructed=r,result=result));save()
  assert r['lower'] is None and r['upper'] is None and not r['gate_rows'] and result['robust_opposite_pair'] is None
  assert result['open_probes'][0]['correct'] and result['open_probes'][0]['robust'] and result['open_probes'][0]['prediction']==0
  A=Arithmetic();x=[1,2,3];L=[[0,0,0],[1,-2,1],[-1,0,0]];result=classify(x,L,[],F(0),A)
  out['rows'].append(dict(name='line_only_tie',x=x,L=L,result=result,counts=dict(A.counts),max_integer_bits=A.bits));save()
  assert result['prediction']==0 and not result['robust'] and result['radius'] is None and result['nonidentical_winner_ties']==[1]
  out['complete']=True;save();print(json.dumps(dict(complete=True,controls=len(out['rows']),cpu=out['cpu'])))
 except Exception as exc:out['failure']=dict(error=repr(exc),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()
