"""Execute only the extracted reporting loop with deterministic fake providers."""
import ast,hashlib,json,resource,signal,time,types
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stop(*args):raise TimeoutError('19CPU soft cap')
def loop(path):
 tree=ast.parse(Path(path).read_text());main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
 found=[x for x in ast.walk(main) if isinstance(x,ast.For) and ast.unparse(x.target)=='(i, m)' and ast.unparse(x.iter)=='enumerate(models)']
 assert len(found)==1
 return compile(ast.fix_missing_locations(ast.Module(body=found,type_ignores=[])),str(path),'exec')

def exercise(code,N,zero=False,changed_output=False):
 events=[]
 class Ops:
  def __init__(self):self.n=0
 class Native:
  def __init__(self):self.STATS=dict(cut=0,resolved=0,cells=0)
  def est75(self,m,ops,**kwargs):
   for k,v in m['stats'].items():self.STATS[k]+=v
   ops.n+=m['cost'];events.append(['est',m['i'],m['rescaled']]);return m['value']
 E=Native()
 def exact(m,ops,st):ops.n=100+m['i'];st['leaves']=50+m['i'];events.append(['truth',m['i']]);return .25+m['i']/128
 def rescale(m,rng):
  events.append(['rescale',m['i']]);return dict(m,rescaled=True,stats={k:(0 if zero else 2*v+3) for k,v in m['stats'].items()},cost=9999,value=m['value']+(1/1024 if changed_output else 0))
 models=[dict(i=i,rescaled=False,stats=dict(cells=0 if zero else i+1,cut=0 if zero else 2*i+3,resolved=0 if zero else i),cost=200+i,value=.5+i/128) for i in range(N)]
 g=dict(models=models,rows=[],inv=0.,E75=E,Ops=Ops,EX=types.SimpleNamespace(exact_acc=exact),REC={},rescale=rescale,rng=object())
 exec(code,g);return dict(rows=g['rows'],inv=g['inv'],events=events,final_stats=E.STATS,inputs=models)

def main():
 resource.setrlimit(resource.RLIMIT_CPU,(19,20));signal.signal(signal.SIGXCPU,stop);dst=HERE/'controls.json';assert not dst.exists();start=time.process_time();out=dict(complete=False,rows=[],new_real_model_estimator_mass_MC_calls=0)
 def save():dst.write_text(json.dumps(out,indent=2)+'\n')
 save()
 try:
  p=json.loads((HERE/'protocol.json').read_text());assert p['source_gate']!='pending'
  for path,h in p['pins'].items():assert sha(path)==h,path
  old=loop(HERE/'frozen/evalH.py');fixed=loop(HERE/'evalH_fixed.py')
  for case in p['cases']:
   out['active']=case['name'];save();a=exercise(old,case['N'],case['zero'],case['changed_output']);b=exercise(fixed,case['N'],case['zero'],case['changed_output'])
   assert a['events']==b['events'] and a['final_stats']==b['final_stats'] and a['inv']==b['inv']
   for i,(x,y,m) in enumerate(zip(a['rows'],b['rows'],a['inputs'],strict=True)):
    for k in ('truth','est','ops','exact_ops','leaves'):assert x[k]==y[k]
    for k,v in m['stats'].items():
     assert y[k]==v
     extra=(0 if case['zero'] else 2*v+3) if i%8==0 else 0
     assert x[k]==v+extra
   nselected=len(range(0,case['N'],8));assert sum(e[0]=='est' for e in a['events'])==case['N']+nselected
   out['rows'].append(dict(case=case,old=a,fixed=b,rescale_calls=nselected,all_checks_pass=True));save()
  out.update(complete=True,active=None,protocol_sha256=sha(HERE/'protocol.json'),source_sha256=sha(__file__),scope='Extracted real reporting-loop code with deterministic fake providers only; source patch preserves calls/estimates/ops/invariance but isolates permodel statistics. No real network, mass/truth evaluator, random or checkpoint call.')
 except BaseException as e:out['failure']=repr(e);raise
 finally:out['cpu_seconds']=time.process_time()-start;save()
 print(json.dumps(dict(complete=out['complete'],cases=len(out['rows']),cpu=out['cpu_seconds'])))
if __name__=='__main__':main()
