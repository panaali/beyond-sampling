import hashlib,itertools,json,resource,signal,time
from pathlib import Path
import count as C
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stop(*args):raise TimeoutError('19CPU soft cap')
def cases():
 return [('single',1,['000'],3,[4,3]),('duplicates',1,['000']*4,3,[4,3]),('last_branch',1,['000','001'],3,[4,3]),('first_branch',1,['000','100'],6,[6,6]),('all_H1T3',1,[''.join(x) for x in itertools.product('01',repeat=3)],10,[10,10]),('all_H1T4',1,[''.join(x) for x in itertools.product('01',repeat=4)],34,[30,30,34]),('all_H2T3',2,[''.join(x) for x in itertools.product('01',repeat=6)],216,[192,216]),('shared_suffix_H2T4',2,[g+'000000' for g in ('00','01','10','11')],144,[114,138,144])]

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(19,20));signal.signal(signal.SIGXCPU,stop);dst=HERE/'controls.json';assert not dst.exists();start=time.process_time();out=dict(complete=False,rows=[],invalid=[],source_sha256=sha(__file__))
 def save():dst.write_text(json.dumps(out,indent=2)+'\n')
 save()
 try:
  protocol=json.loads((HERE/'protocol.json').read_text())
  for p,h in protocol['pins'].items():assert sha(p)==h,p
  for name,H,words,baseline,cuts in cases():
   out['active']=name;save();r=C.census(words,H);assert C.total(r['baseline'])==baseline
   assert [C.total(c['combined']) for c in r['cuts']]==cuts
   out['rows'].append(dict(name=name,result=r,expected_baseline=baseline,expected_combined=cuts));save()
  for name,words,H in [('empty',[],1),('ragged',['000','00'],1),('nonbinary',['002'],1),('invalidH',['000'],0)]:
   out['active']=name;save()
   try:C.census(words,H)
   except ValueError as e:out['invalid'].append(dict(name=name,status='rejected',reason=str(e)))
   else:raise AssertionError('accepted invalid '+name)
   save()
  out.update(complete=True,active=None,protocol_sha256=sha(HERE/'protocol.json'))
 except BaseException as e:out['failure']=repr(e);raise
 finally:out['cpu_seconds']=time.process_time()-start;save()
 print(json.dumps(dict(complete=out['complete'],controls=len(out['rows']),invalid=len(out['invalid']),cpu=out['cpu_seconds'])))
if __name__=='__main__':main()
