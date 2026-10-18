"""Frozen one-target readout predicate; standard-library stored-array decoding only."""
from pathlib import Path
from fractions import Fraction as F
import ast,hashlib,json,math,os,resource,struct,subprocess,time,traceback,zipfile
from predicate import readout_predicate
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'target_result.json';PROGRESS=ROOT/'target_progress.jsonl'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(d):
 p=OUT.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');os.replace(p,OUT)
def event(d):
 with PROGRESS.open('a') as f:f.write(json.dumps(d,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def main():
 assert not OUT.exists() and not PROGRESS.exists() and not OUT.with_suffix('.tmp').exists()
 pp=ROOT/'target_protocol.json';gp=ROOT/'target_gate.json';p=json.loads(pp.read_text());g=json.loads(gp.read_text())
 assert g['complete'] and g['execution_allowed'] and g['protocol_sha256']==sha(pp)
 assert {str(ROOT/x) for x in ('target.py','predicate.py','TARGET_PROTOCOL.md','control_protocol.json','controls_result.json')}<=set(p['pins'])
 assert set(g['reviews'])=={'controls_saved','target_source'}
 for role,q in g['reviews'].items():
  assert g['pins'][q['path']]==q['sha256']==sha(q['path'])
  d=json.loads(Path(q['path']).read_text());assert d['complete'] is True and d['pass_all'] is True
  assert q['commit'] in g['published_commits']
 for path,h in {**p['pins'],**g['pins']}.items():assert sha(path)==h,(path,'pin')
 for commit in g['published_commits']:subprocess.run(['git','merge-base','--is-ancestor',commit,'origin/ccr-6a61f25c-wn80ii'],check=True)
 for kind,cap in ((resource.RLIMIT_CPU,20),(resource.RLIMIT_AS,2097152*1024)):
  soft,hard=resource.getrlimit(kind);assert 0<soft<=cap and 0<hard<=cap
 assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
 start=time.process_time();out={'task':'T229_target','complete':False,'pass_all':False,'protocol_sha256':sha(pp),'gate_sha256':sha(gp),'checkpoint_decodes':0,'model_calls':0,'geometry_calls':0,'mass_calls':0,'random_calls':0};save(out);event({'phase':'start'})
 try:
  old=json.loads(Path(p['prior_result']).read_text());rv=json.loads(Path(p['prior_saved_review']).read_text())
  assert old['status']=='complete' and rv['complete'] is True and rv['pass_all'] is True
  assert old['weights']['archive_sha256']==p['pins'][p['archive']]
  assert old['weights']['checkpoint_sha256']==p['checkpoint_sha256']
  meta=old['archive_arrays']['Wo'];assert meta['shape']==[10,16] and meta['dtype']=='float64'
  event({'phase':'decode_attempt'});save(out)
  with zipfile.ZipFile(p['archive']) as z:
   assert sorted(z.namelist())==['Wh.npy','Wi.npy','Wo.npy']
   info=z.getinfo('Wo.npy');assert info.file_size<=65536
   raw=z.read('Wo.npy')
  assert raw[:6]==b'\x93NUMPY';version=tuple(raw[6:8]);assert version in ((1,0),(2,0))
  sizebytes=2 if version==(1,0) else 4;hlen=int.from_bytes(raw[8:8+sizebytes],'little');offset=8+sizebytes
  assert 0<hlen<=65536 and offset+hlen<=len(raw)
  header=ast.literal_eval(raw[offset:offset+hlen].decode('latin1'))
  assert header=={'descr':'<f8','fortran_order':False,'shape':(10,16)},header
  payload=raw[offset+hlen:];assert len(payload)==1280
  assert hashlib.sha256(payload).hexdigest()==meta['sha256']==p['wo_payload_sha256']
  values=struct.unpack('<160d',payload);assert all(math.isfinite(x) for x in values)
  ratios=[x.as_integer_ratio() for x in values];rows=[[F(*q) for q in ratios[i*16:(i+1)*16]] for i in range(10)]
  out['readout']={'shape':[10,16],'archive_sha256':sha(p['archive']),'npy_version':version,'header':{'descr':header['descr'],'fortran_order':header['fortran_order'],'shape':header['shape']},'payload_sha256':meta['sha256'],'payload_hex':payload.hex(),'rows':[[str(x) for x in row] for row in rows],
   'negative_zero_count':sum(x==0 and math.copysign(1,x)<0 for x in values),'inherited_checkpoint_sha256':p['checkpoint_sha256']}
  save(out);event({'phase':'decode_result','coefficients':160})
  assert time.process_time()-start<16
  event({'phase':'predicate_attempt'});save(out);out['predicate']=readout_predicate(rows);save(out)
  assert out['predicate']['rows']==10 and out['predicate']['width']==16 and len(out['predicate']['pair_certificates'])==120
  assert out['predicate']['status'] in ('identity_forced_by_readout','inconclusive')
  event({'phase':'predicate_result','status':out['predicate']['status']})
  assert len(PROGRESS.read_text().splitlines())==5 and time.process_time()-start<16
  out['complete']=out['pass_all']=True;out['cpu_seconds']=time.process_time()-start;out['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  save(out);event({'phase':'complete'})
 except BaseException as ex:
  out['failure']={'type':type(ex).__name__,'message':str(ex),'traceback':traceback.format_exc()};out['cpu_seconds']=time.process_time()-start;save(out);event({'phase':'failure','type':type(ex).__name__});raise
if __name__=='__main__':main()
