"""Finite observer coverage audit; no model, geometry or rank evaluation."""
from pathlib import Path
import argparse,ctypes as ct,hashlib,json,os,subprocess,sys,time
START=time.process_time()
HERE=Path(__file__).resolve().parent
CENSUS=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_runtime_census')
LIBRARY=Path('/workspace/clean-neurons/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f48b354e.so')

def child(mode):
 import numpy as np
 lib=ct.CDLL(str(LIBRARY));spy=ct.CDLL(str(CENSUS/'spy.so')) if mode=='observed' else None
 fn=(spy if spy is not None else lib).scipy_dlarfg_64_;ip=ct.POINTER(ct.c_int64);dp=ct.POINTER(ct.c_double);fn.argtypes=[ip,dp,dp,ip,dp];fn.restype=None
 if spy:
  spy.census_size.restype=ct.c_int64;spy.census_overflow.restype=ct.c_int64
 rows=[]
 for n in [1,2,4]:
  for alpha in [1.,1e-310]:
   for tiny in [False,True]:
    x=np.array([(1+j/8)*(1e-310 if tiny else 1.) for j in range(max(1,n-1))]);a=ct.c_double(alpha);tau=ct.c_double(-123);nn=ct.c_int64(n);inc=ct.c_int64(1)
    if spy:spy.census_reset()
    fn(ct.byref(nn),ct.byref(a),x.ctypes.data_as(dp),ct.byref(inc),ct.byref(tau))
    out=dict(n=n,alpha=alpha,tiny=tiny,values=[a.value,tau.value]+x.tolist(),output_hex=[float(v).hex() for v in [a.value,tau.value]+x.tolist()])
    if spy:
     size=spy.census_size();assert 0<=size<=1024 and spy.census_overflow()==0
     buf=(ct.c_int64*(size*6))();spy.census_get(buf);out['records']=[list(buf[i*6:i*6+6]) for i in range(size)]
    rows.append(out)
 result=dict(mode=mode,rows=rows,cpu=time.process_time()-START)
 (HERE/f'controls_{mode}.json').write_text(json.dumps(result,indent=2)+'\n')

def saved():
 out=[]
 for mode in ['exact','recipe108']:
  for i in range(8):
   d=json.loads((CENSUS/f'{mode}_{i}.json').read_text())
   for phase in ['native','cold']:
    q=d[phase];rows=q['kernels'];active=sum(r['calls'] for r in rows if r['kernel']=='dlarfg' and r['m']>1);norm=sum(r['calls'] for r in rows if r['kernel']=='dnrm2')
    assert q['overflow']==0 and all(r['calls']>0 and r['failures']==0 for r in rows)
    out.append(dict(mode=mode,batch=i,phase=phase,active_reflectors=active,norm_calls=norm,equal=active==norm,max_counter=max(r['calls'] for r in rows),rows=len(rows)))
 return out

def main():
 proto=json.loads((HERE/'protocol.json').read_text())
 for f,h in proto['binary_pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
 env=dict(os.environ);env.pop('LD_PRELOAD',None);env.pop('LD_DEBUG',None)
 subprocess.run([sys.executable,__file__,'--mode','direct'],env=env,check=True)
 env.update(LD_PRELOAD=str(CENSUS/'spy.so'),LD_DEBUG='bindings')
 run=subprocess.run([sys.executable,__file__,'--mode','observed'],env=env,check=True,stderr=subprocess.PIPE)
 (HERE/'binding_trace.log').write_bytes(run.stderr)
 relevant=[s for s in run.stderr.decode().splitlines() if 'scipy_dnrm2_64_' in s or 'scipy_dlarfg_64_' in s]
 (HERE/'binding_lines.txt').write_text('\n'.join(relevant)+'\n')
 a=json.loads((HERE/'controls_direct.json').read_text());b=json.loads((HERE/'controls_observed.json').read_text());cases=[]
 for x,y in zip(a['rows'],b['rows']):
  assert x['output_hex']==y['output_hex'];rs=y['records'];nc=sum(r[4] for r in rs if r[0]==9);pc=sum(r[4] for r in rs if r[0]==8)
  expected=0 if x['n']<=1 else (2 if x['alpha']==1e-310 and x['tiny'] else 1)
  cases.append(dict(n=x['n'],alpha=x['alpha'],tiny=x['tiny'],parity=True,parent_calls=pc,norm_calls=nc,expected_norm=expected));assert pc==1 and nc==expected
 phases=saved();assert all(r['equal'] for r in phases)
 result=dict(controls=cases,saved_phases=phases,all_equal=True,binding_lines=relevant,pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CENSUS/'spy.c',CENSUS/'spy.so',CENSUS/'observer.py',CENSUS/'run.py',HERE/'check.py']},direct_cpu=a['cpu'],observed_cpu=b['cpu'],parent_cpu=time.process_time()-START,scope='Finite control evidence plus installed static PLT callsites and reviewed saved phase checks. Not a generic no-rescale claim; historical observer/source identity remains a provenance premise.')
 (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['controls','saved_phases','pins','binding_lines']}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--mode');args=ap.parse_args();child(args.mode) if args.mode else main()
