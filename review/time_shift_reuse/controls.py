"""Frozen exact synthetic controls; no trained gate enumeration or mass calls."""
import hashlib,itertools,json,resource,signal,time
from fractions import Fraction as F
from pathlib import Path
from collections import Counter
import template as T
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dot(a,b):return sum((x*y for x,y in zip(a,b,strict=True)),F(0))
def timeout(*args):raise TimeoutError('19CPU soft cap')
def direct(W,w,word,z,masked):
 H=len(w);h=tuple(z[:H]);values=[];actual=[]
 for x,g in zip(z[H:],word,strict=True):
  pre=tuple(dot(row,h)+v*x for row,v in zip(W,w,strict=True));values.extend(pre);actual.append(tuple(int(v>0) for v in pre))
  h=tuple(v if on else F(0) for v,on in zip(pre,g,strict=True)) if masked else tuple(max(v,F(0)) for v in pre)
 return h,tuple(values),tuple(actual)

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(19,20));signal.signal(signal.SIGXCPU,timeout)
 dst=HERE/'controls.json';progress=HERE/'progress.jsonl';assert not dst.exists() and not progress.exists();start=time.process_time()
 out=dict(complete=False,rows=[],counterexamples=[],source_sha256=sha(__file__),counts=dict(words=0,direct_masked=0,direct_relu=0,shift_checks=0,composition_checks=0),new_trained_geometry_mass_MC_calls=0)
 def save():dst.write_text(json.dumps(T.pack(out),indent=2)+'\n')
 def record(row):
  with progress.open('a') as f:f.write(json.dumps(T.pack(row))+'\n')
 save()
 try:
  protocol=json.loads((HERE/'protocol.json').read_text())
  for p,h in protocol['pins'].items():assert sha(p)==h,p
  assert protocol['word_lengths']==[1,2,3] and protocol['shifts']==[0,2]
  for model in protocol['models']:
   W=tuple(tuple(F(x) for x in row) for row in model['Wh']);w=tuple(F(x) for x in model['Wi']);H=len(w);assert H==2
   for k in (1,2,3):
    for bits in itertools.product((0,1),repeat=H*k):
     word=tuple(tuple(bits[H*t:H*(t+1)]) for t in range(k));key=model['name']+':'+''.join(map(str,bits));record(dict(stage='started',key=key))
     c=T.build(W,w,word);d=H+k
     vectors=[(F(0),)*d]+[tuple(F(sign if j==q else 0) for j in range(d)) for q in range(d) for sign in (1,-1)]+[(F(1),)*d,tuple(F((j+1)*(-1 if j%2 else 1)) for j in range(d))]
     checks=[];setup=Counter(c['counts']);peak=list(c['bits'])
     for z in vectors:
      actual,pre,gate_word=direct(W,w,word,z,True);out['counts']['direct_masked']+=1
      mapped=tuple(dot(row,z) for row in c['output']);mapped_pre=tuple(dot(row,z) for row,on in c['planes']);assert actual==mapped and pre==mapped_pre
      compatible=all((v>0) if on else (v<=0) for v,(row,on) in zip(pre,c['planes'],strict=True))
      real,_,real_word=direct(W,w,word,z,False);out['counts']['direct_relu']+=1
      if compatible:assert real==actual and real_word==word
      for shift in (0,2):
       shifted=T.shifted(c,shift,k+2);ambient=[F(13+j) for j in range(H+k+2)];ambient[:H]=z[:H];ambient[H+shift:H+shift+k]=z[H:]
       assert tuple(dot(row,ambient) for row in shifted['output'])==actual
       assert tuple(dot(row,ambient) for row,on in shifted['planes'])==pre
       out['counts']['shift_checks']+=1
      checks.append(dict(vector=z,masked=actual,preactivation=pre,compatible=compatible,direct_relu=real,direct_gate_word=real_word,nonzero_form_zero_values=sum(bool(any(row)) and v==0 for v,(row,on) in zip(pre,c['planes'],strict=True))))
     for split in range(1,k):
      left=T.build(W,w,word[:split]);right=T.build(W,w,word[split:]);joined=T.compose(left,right)
      assert joined['output']==c['output'] and joined['planes']==c['planes'] and joined['impossible_identically_zero_on']==c['impossible_identically_zero_on']
      for a in (left,right,joined):setup.update(a['counts']);peak=[max(x,y) for x,y in zip(peak,a['bits'],strict=True)]
      out['counts']['composition_checks']+=1
     row=dict(key=key,model=model['name'],word=word,template=c,checks=checks,control_template_and_composition_counts=dict(setup),peak_reduced_fraction_bits=peak)
     out['rows'].append(row);out['counts']['words']+=1;record(dict(stage='complete',row=row))
  assert out['counts']==dict(words=420,direct_masked=5220,direct_relu=5220,shift_checks=10440,composition_checks=720)
  record(dict(stage='counterexamples_started'))
  x=F(-3,2);hs=(F(1),F(2));pre=[h+x for h in hs];assert pre[0]<0<pre[1]
  out['counterexamples'].append(dict(kind='same_word_formula_different_incoming_domain',incoming=hs,input=x,preactivation=pre,actual_gates=[0,1]))
  xs=[(F(3),F(2),F(1)),(F(2),F(3),F(1))];histories=[];states=[];truth=[]
  for seq in xs:
   h=F(0);history=[]
   for x in seq:h=max(x,F(0));history.append(int(x>0))
   histories.append(history);states.append(h);truth.append(sorted(range(3),key=lambda j:seq[j])[-2])
  assert histories[0]==histories[1] and states==[F(1),F(1)] and truth==[1,0]
  out['counterexamples'].append(dict(kind='same_terminal_state_different_truth',inputs=xs,Wh=F(0),Wi=F(1),fixed_output=0,histories=histories,states=states,truth=truth,correct=[False,True]))
  record(dict(stage='counterexamples_complete',rows=out['counterexamples']))
  out.update(complete=True,protocol_sha256=sha(HERE/'protocol.json'),progress_sha256=sha(progress),scope='Conditional affine/gate algebra only. Synthetic negative incoming coordinates are algebra controls, not recurrence reachability claims. Infeasible words retained, no hit-rate/probability/speed inference. Template arithmetic named separately; all validation/control/bit costs remain real research work.')
 except BaseException as e:out['failure']=repr(e);record(dict(stage='failure',failure=repr(e)));raise
 finally:out.update(cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save()
 print(json.dumps(dict(complete=out['complete'],counts=out['counts'],cpu=out['cpu_seconds'])))

if __name__=='__main__':main()
