from pathlib import Path
from fractions import Fraction as F
import hashlib,json,resource,time
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h4_4/m16/approaches/weight_circles/common_mode');O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dec(v):
 if isinstance(v,list):
  if len(v)==2 and all(isinstance(x,str) for x in v):return F(*map(int,v))
  return [dec(x) for x in v]
 if isinstance(v,dict):return {k:dec(x) for k,x in v.items()}
 return v
def dot(a,b):return sum(x*y for x,y in zip(a,b,strict=True))
def probe(x,L,gates,p):
 y=[z+p['c'] for z in x];values=[dot(row,y) for row in L];winner=min(j for j,v in enumerate(values) if v==max(values));target=sorted(range(len(x)),key=lambda j:x[j])[-2]
 assert p['y']==y and p['values']==values and p['prediction']==winner and p['target']==target and p['correct']==(winner==target)
 ties=[j for j in range(len(L)) if j!=winner and values[j]==values[winner] and L[j]!=L[winner]]
 assert p['nonidentical_winner_ties']==ties and p['robust']==(not ties)
 for g in gates:assert dot(g['row'],y)!=0 and (dot(g['row'],y)>0)==g['active']
 if ties:assert p['radius'] is None;return
 radii=[F(1)]+[abs(dot(g['row'],y))/(2*sum(map(abs,g['row']))) for g in gates]
 radii += [F(abs(x[i]-x[j]),4) for i in range(len(x)) for j in range(i)]
 radii += [F(values[winner]-values[j],2*sum(abs(a-b) for a,b in zip(L[winner],L[j]))) for j in range(len(L)) if L[winner]!=L[j]]
 assert p['radius']==min(radii)>0

def validate(x,r,q):
 L=r['L'];gates=r['gate_rows'];lo=hi=None
 for g in gates:
  v=dot(g['row'],x);s=sum(g['row']);assert g['intercept']==v and g['slope']==s and v!=0 and g['active']==(v>0)
  if s:
   z=F(-v,s)
   if (s>0)==g['active']:lo=z if lo is None else max(lo,z)
   else:hi=z if hi is None else min(hi,z)
 assert r['lower']==q['lower']==lo and r['upper']==q['upper']==hi
 a=[dot(row,x) for row in L];b=[sum(row) for row in L]
 assert q['logit_intercepts']==a and q['logit_slopes']==b
 roots=sorted({F(a[j]-a[i],b[i]-b[j]) for i in range(len(L)) for j in range(i) if b[i]!=b[j] and (lo is None or F(a[j]-a[i],b[i]-b[j])>lo) and (hi is None or F(a[j]-a[i],b[i]-b[j])<hi)})
 assert roots==q['roots'];cuts=[lo]+roots+[hi]
 assert len(q['open_probes'])==len(roots)+1 and len(q['crossing_points'])==len(roots)
 for p,left,right in zip(q['open_probes'],cuts[:-1],cuts[1:],strict=True):
  assert p['kind']=='open_interval' and p['lower']==left and p['upper']==right and (left is None or left<p['c']) and (right is None or p['c']<right)
  probe(x,L,gates,p)
 for p,c in zip(q['crossing_points'],roots,strict=True):assert p['c']==c;probe(x,L,gates,p)
 candidates=[[i,j] for i,p in enumerate(q['open_probes']) for j,z in enumerate(q['open_probes']) if j>i and p['robust'] and z['robust'] and p['correct']!=z['correct']]
 assert q['robust_opposite_pair']==(candidates[0] if candidates else None)
 assert q['line_open_change']==(len({p['correct'] for p in q['open_probes']})>1)

def main():
 g=json.loads((P/'controls_gate.json').read_text());r=dec(json.loads((P/'controls.json').read_text()));assert g['complete'] and r['complete'] and r['gate_sha256']==sha(P/'controls_gate.json')
 for p,h in g['pins'].items():assert sha(Path(p))==h,p
 assert r['new_trained_forward_calls']==r['new_reference_draws']==0
 assert [a['name'] for a in r['rows']]==['crossing','invariant','zero_permanent_tie','line_only_tie']
 for row in r['rows'][:3]:validate(row['x'],row['reconstructed'],row['result'])
 row=r['rows'][3];probe(row['x'],row['L'],[],row['result'])
 assert r['rows'][0]['reconstructed']['L']==[[-2,0,1],[0,0,0],[-4,-2,-1]]
 assert r['rows'][1]['reconstructed']['L']==[[1,1,1],[0,0,0],[-4,-2,-1]]
 assert r['rows'][2]['reconstructed']['L']==[[0,0,0]]*3
 assert r['rows'][0]['result']['robust_opposite_pair']==[0,1] and r['rows'][2]['result']['open_probes'][0]['robust'] and not row['result']['robust']
 result=dict(complete=True,controls=4,author_sha256=sha(P/'controls.json'),author_cpu=r['cpu'],review_cpu=time.process_time()-start,scope='Independent exact intervals, complete pair roots, probe logits/first-index labels/ties/radii and open-change flags. Source algebra checks model matrices. No author control/evaluator replay; named operation ledger is source convention, not full cost.')
 (O/'t191_common_mode_controls_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
