"""Fixed T229 analytic controls; no target or imported author evaluator."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,os,resource,subprocess,time,traceback
from predicate import readout_predicate
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'controls_result.json';PROGRESS=ROOT/'controls_progress.jsonl'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def serial(x):
 if isinstance(x,F):return {'rational':str(x)}
 if isinstance(x,(list,tuple)):return [serial(y) for y in x]
 if isinstance(x,complex):return {'complex':[x.real,x.imag]}
 if isinstance(x,float) and not math.isfinite(x):return {'float':str(x)}
 return x
def save(obj):
 p=OUT.with_suffix('.tmp');p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(p,OUT)
def event(obj):
 with PROGRESS.open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def fixtures():
 I='identity_forced_by_readout';U='inconclusive'
 return [
 ('identity',[[1,0],[0,1]],I),('positive_parallel',[[1,2],[2,4]],U),
 ('negative_parallel',[[1,-2],[2,-4]],I),('zero_column',[[1,0],[2,0]],U),
 ('all_zero',[[0,0],[0,0]],U),('single_nonzero',[[-3]],I),('single_zero',[[0]],U),
 ('cycle_equal_readout',[[1,1,1]],U),('near_parallel_dyadic',[[1,1],[1,1+F(1,2**80)]],I),
 ('three_distinct',[[1,0,1],[0,1,1]],I),('positive_gauge_columns',[[F(1,2),0,3],[0,2,3]],I),
 ('invertible_row_transform',[[1,1],[1,-1]],I),
 ('empty',[],ValueError),('zero_width',[[]],ValueError),('ragged',[[1],[1,2]],ValueError),
 ('scalar_row',[1],TypeError),('float_coefficient',[[1.0]],TypeError),
 ('nonfinite',[[float('inf')]],TypeError),('text_coefficient',[['1/2']],TypeError),
 ('bool_coefficient',[[True]],TypeError)]
def main():
 assert not OUT.exists() and not PROGRESS.exists() and not OUT.with_suffix('.tmp').exists()
 p=ROOT/'control_protocol.json';gate_path=ROOT/'control_gate.json';protocol=json.loads(p.read_text());gate=json.loads(gate_path.read_text())
 assert gate['complete'] and gate['execution_allowed'] and gate['protocol_sha256']==sha(p)
 assert {'predicate.py','controls.py','PROOF.md','CONTROL_PROTOCOL.md'}<=set(protocol['pins'])
 for rel,h in protocol['pins'].items():assert sha(ROOT/rel)==h
 for path,h in gate['pins'].items():assert sha(Path(path))==h
 review=json.loads(Path(gate['review']['path']).read_text());assert review['complete'] is True and review['pass_all'] is True
 assert gate['pins'][gate['review']['path']]==gate['review']['sha256'] and gate['review']['commit'] in gate['published_commits']
 for commit in gate['published_commits']:subprocess.run(['git','merge-base','--is-ancestor',commit,'origin/ccr-6a61f25c-wn80ii'],check=True)
 for kind,cap in ((resource.RLIMIT_CPU,20),(resource.RLIMIT_AS,2097152*1024)):
  soft,hard=resource.getrlimit(kind);assert 0<soft<=cap and 0<hard<=cap
 assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
 start=time.process_time();out={'task':'T229_controls','complete':False,'pass_all':False,'rows':[],'protocol_sha256':sha(p),'gate_sha256':sha(gate_path),'target_model_calls':0};save(out);event({'phase':'start'})
 try:
  for name,rows,expect in fixtures():
   rec={'name':name,'input':serial(rows),'expected':expect if isinstance(expect,str) else expect.__name__}
   out['rows'].append(rec);save(out);event({'phase':'attempt','name':name})
   try:result=readout_predicate(rows)
   except (ValueError,TypeError) as ex:
    rec['exception']={'type':type(ex).__name__,'message':str(ex)};save(out)
    assert isinstance(expect,type) and type(ex) is expect
   else:
    rec['result']=result;save(out);assert isinstance(expect,str) and result['status']==expect
    assert len(result['pair_certificates'])==len(rows[0])*(len(rows[0])-1)//2
   rec['pass']=True;save(out);event({'phase':'result','name':name,'pass':True});assert time.process_time()-start<16
  assert len(out['rows'])==20 and len(PROGRESS.read_text().splitlines())==41
  out['complete']=out['pass_all']=True;out['cpu_seconds']=time.process_time()-start;out['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;save(out);event({'phase':'complete'})
 except BaseException as ex:
  out['failure']={'type':type(ex).__name__,'message':str(ex),'traceback':traceback.format_exc()};out['cpu_seconds']=time.process_time()-start;save(out);event({'phase':'failure','type':type(ex).__name__});raise
if __name__=='__main__':main()
