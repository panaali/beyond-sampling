"""Frozen saved-only census; execute only after completed control review."""
import hashlib,json,resource,signal,time
from pathlib import Path
import count as C
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stop(*args):raise TimeoutError('19CPU soft cap')
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(19,20));signal.signal(signal.SIGXCPU,stop);dst=HERE/'results.json';assert not dst.exists();start=time.process_time();out=dict(complete=False,source_sha256=sha(__file__),new_model_geometry_mass_MC_calls=0)
 def save():dst.write_text(json.dumps(out,indent=2)+'\n')
 save()
 try:
  p=json.loads((HERE/'protocol.json').read_text());gate=json.loads((HERE/'run_scope.json').read_text())
  assert gate['controls_review'] and gate['source_gate']
  for path,h in p['pins'].items():assert sha(path)==h,path
  assert sha(HERE/'controls.json')==gate['controls_sha256']
  ctl=json.loads((HERE/'controls.json').read_text());assert ctl['complete'] and len(ctl['rows'])==8 and len(ctl['invalid'])==4
  att=json.loads(Path(p['attribution']).read_text());assert att['complete'] and att['source_sha256']==sha(p['geometry'])
  assert att['checkpoint_sha256']==p['checkpoint_sha256']
  words=att['sets']['gate']['history_keys'];assert len(set(words))==450 and set(words)==set(att['histories'])
  out.update(stage='validated_pins',protocol_sha256=sha(HERE/'protocol.json'),run_scope_sha256=sha(HERE/'run_scope.json'),checkpoint_sha256=att['checkpoint_sha256']);save()
  out['census']=C.census(words,4);assert out['census']['T']==4
  assert [r['cut'] for r in out['census']['cuts']]==[1,2,3]
  out.update(complete=True,stage='complete',scope=p['scope'])
 except BaseException as e:out['failure']=repr(e);raise
 finally:out.update(cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save()
 print(json.dumps(dict(complete=out['complete'],histories=len(out['census']['unique_histories']),baseline=C.total(out['census']['baseline']),cuts=[dict(cut=r['cut'],warm=C.total(r['warm']),cold=C.total(r['cold']),combined=C.total(r['combined'])) for r in out['census']['cuts']],cpu=out['cpu_seconds'])))
if __name__=='__main__':main()
