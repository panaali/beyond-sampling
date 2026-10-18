"""Independent exact IEEE payload / saved pair certificates. No author imports."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations
import ast,hashlib,json,time,zipfile
B=Path('/workspace/work/recover-small_h2_34/review/weight_automorphism')
O=Path(__file__).with_name('t229_target_saved_review.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ieee(b):
 n=int.from_bytes(b,'little');s=-1 if n>>63 else 1;e=(n>>52)&2047;m=n&((1<<52)-1)
 assert e!=2047
 if e:m+=1<<52
 power=(e-1023-52) if e else -1074
 return s*F(m)*F(2)**power,n==1<<63
def main():
 assert not O.exists();start=time.process_time()
 p=read(B/'target_protocol.json');g=read(B/'target_gate.json');r=read(B/'target_result.json');q=r['readout'];a=r['predicate']
 for mapping in (p['pins'],g['pins']):
  for path,h in mapping.items():assert sha(path)==h
 assert r['complete'] is True and r['pass_all'] is True and g['complete'] is True and g['execution_allowed'] is True
 assert r['protocol_sha256']==g['protocol_sha256']==sha(B/'target_protocol.json') and r['gate_sha256']==sha(B/'target_gate.json')
 assert set(g['reviews'])=={'controls_saved','target_source'}
 for role,v in g['reviews'].items():
  rv=read(v['path']);assert rv['complete'] is True and rv['pass_all'] is True and v['sha256']==g['pins'][v['path']] and v['commit'] in g['published_commits']
 prior=read(p['prior_result']);review=read(p['prior_saved_review'])
 assert prior['status']=='complete' and review['complete'] is True and review['pass_all'] is True
 assert prior['weights']['checkpoint_sha256']==p['checkpoint_sha256']==q['inherited_checkpoint_sha256']
 assert prior['weights']['archive_sha256']==p['pins'][p['archive']]==q['archive_sha256']
 meta=prior['archive_arrays']['Wo'];assert meta['shape']==[10,16] and meta['dtype']=='float64'
 with zipfile.ZipFile(p['archive']) as z:
  assert sorted(z.namelist())==['Wh.npy','Wi.npy','Wo.npy'];raw=z.read('Wo.npy')
 assert raw[:8]==b'\x93NUMPY\x01\x00';h=int.from_bytes(raw[8:10],'little');header=ast.literal_eval(raw[10:10+h].decode('latin1'));data=raw[10+h:]
 assert header=={'descr':'<f8','fortran_order':False,'shape':(10,16)} and len(data)==1280
 assert data.hex()==q['payload_hex'] and hashlib.sha256(data).hexdigest()==q['payload_sha256']==p['wo_payload_sha256']==meta['sha256']
 assert q['shape']==[10,16] and q['npy_version']==[1,0] and q['header']=={'descr':'<f8','fortran_order':False,'shape':[10,16]}
 vals=[ieee(data[j:j+8]) for j in range(0,1280,8)];assert sum(z for _,z in vals)==q['negative_zero_count']
 rows=[[v for v,_ in vals[j:j+16]] for j in range(0,160,16)];assert [[F(x) for x in row] for row in q['rows']]==rows
 cols=list(zip(*rows));assert [[F(x) for x in c] for c in a['columns']]==[list(c) for c in cols]
 assert a['rows']==10 and a['width']==16 and all(any(c) for c in cols)
 for c,can in zip(cols,a['positive_canonical_columns'],strict=True):
  first=next(x for x in c if x);v=list(map(F,can));assert next(x for x in v if x) in (-1,1)
  assert all(vv*abs(first)==x for vv,x in zip(v,c,strict=True))
 certs=a['pair_certificates'];assert len(certs)==120;minor_count=0
 for cert,(i,j) in zip(certs,combinations(range(16),2),strict=True):
  assert cert['pair']==[i,j] and cert['kind']=='nonzero_minor';u,v=cols[i],cols[j]
  # Independently count lexicographic source sites and validate first nonzero witness.
  found=False
  for x,y in combinations(range(10),2):
   det=u[x]*v[y]-u[y]*v[x];minor_count+=1
   if det:
    assert cert['rows']==[x,y] and F(cert['determinant'])==det;found=True;break
  assert found
 assert a['zero_columns']==[] and a['positive_proportional_classes']==[]
 assert a['status']=='identity_forced_by_readout' and a['partial_setup_counts']=={'normalization_divisions':160,'minor_evaluations':minor_count}
 assert a['not_complete_native_or_bit_cost'] is True
 for key in ('checkpoint_decodes','model_calls','geometry_calls','mass_calls','random_calls'):assert r[key]==0
 events=[json.loads(x) for x in (B/'target_progress.jsonl').read_text().splitlines()]
 assert events==[{'phase':'start'},{'phase':'decode_attempt'},{'phase':'decode_result','coefficients':160},{'phase':'predicate_attempt'},{'phase':'predicate_result','status':'identity_forced_by_readout'},{'phase':'complete'}]
 pins={str(B/n):sha(B/n) for n in ('target_result.json','target_progress.jsonl','target.py','target_protocol.json','target_gate.json','predicate.py','GLOBAL_RELU_COROLLARY.md')}
 out={'complete':True,'pass_all':True,'task':'T229 independent target saved certificate review','assignment':'0d166612','pins':pins,'dependency_pins':p['pins'],'checker_sha256':sha(Path(__file__)),'coefficients':160,'nonzero_minor_certificates':120,'normalization_divisions':160,'minor_evaluations':minor_count,'progress_events':6,'identity_forced_by_readout':True,'review_cpu_seconds':time.process_time()-start,'author_cpu_seconds':r['cpu_seconds'],'new_author_model_checkpoint_calls':0,'method':'Independent integer IEEE754 payload decode, inherited T219 archive/provenance binding, exact saved canonical/minor certificates and source-expression count reconstruction.','corollary':'An invertible linear A commuting with ReLU on all ambient vectors maps the orthant onto itself, including via its inverse; orthant extreme rays imply positive monomial A. Combined with readout certificates, identity is forced under unchanged-logit global weight automorphism premises.','limitations':['Inherited checkpoint-to-archive provenance reused; checkpoint not decoded again.','Universal ReLU commutation and unchanged weights/logits are essential; reachable-only, argmax/common-shift, approximate and noninvertible alternatives are not excluded.','No geometry, accuracy, enumeration cost, or general aggregation impossibility inference.','Partial setup ledger excludes complete native arithmetic and bit costs.']}
 O.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'complete':True,'sha256':sha(O),'cpu':out['review_cpu_seconds']}))
if __name__=='__main__':main()
