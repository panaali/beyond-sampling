"""Frozen product controls, no model/geometry/MC calls or adaptive orders."""
import hashlib,json,math,resource,time,warnings
from fractions import Fraction as F
from pathlib import Path
import mpmath as mp
import numpy as np
from loader import load,Ops
HERE=Path(__file__).resolve().parent

def fixtures(ks):
    rows=[{'name':'positive5orthant','rays':np.eye(5),'ideal':None,'wide':False}]
    for k in ks:
        t=F(1,2**k);c=(1-t*t)/(1+t*t);s=2*t/(1+t*t)
        assert c*c+s*s==1
        for wide in [False,True]:
            R=np.eye(5);R[1,:2]=[float(-c if wide else c),float(s)]
            rows.append({'name':('wide' if wide else 'narrow')+str(k),'rays':R,'ideal':{'tan_half':str(t),'cos':str(-c if wide else c),'sin':str(s)},'wide':wide})
    return rows

def reference(row,dps):
    if row['ideal'] is None:return '0.03125',mp.mpf(1)/32
    c,s=map(F,row['rays'][1,:2])
    with mp.workdps(dps):
        cc=mp.mpf(c.numerator)/c.denominator;ss=mp.mpf(s.numerator)/s.denominator
        val=mp.atan2(ss,cc)/(16*mp.pi)
        return mp.nstr(val,dps),+val

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,22));dest=HERE/'results.json';assert not dest.exists()
    p=json.loads((HERE/'protocol.json').read_text());pins=json.loads((HERE/'pins.json').read_text())
    for path,sha in pins.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
    start=time.process_time();out={'complete':False,'protocol':p,'fixtures':[],'setup':[],'rows':[],'pins_sha256':hashlib.sha256((HERE/'pins.json').read_bytes()).hexdigest(),'numpy_version':np.__version__,'mpmath_version':mp.__version__}
    def save():
        out.update(cpu=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        dest.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    try:
        data=fixtures(p['ks'])
        for row in data:
            s80,v80=reference(row,80);s120,v120=reference(row,120)
            with mp.workdps(140):assert abs(v80-v120)<mp.mpf('1e-70')
            rr=row['rays'];dot=sum(F(rr[0,j])*F(rr[1,j]) for j in range(5));c,s=map(F,rr[1,:2]);det=s if row['ideal'] else F(1)
            out['fixtures'].append({'name':row['name'],'rays_hex':[[float(x).hex() for x in r] for r in rr],'ideal':row['ideal'],'exact_stored_determinant':str(det),'exact_pair_dot':str(dot),'stored_second_ray_norm2':str(c*c+s*s),'reference80':s80,'reference120':s120})
            row['reference120']=s120
        for j in range(1,len(data),2):
            assert out['fixtures'][j]['exact_stored_determinant']==out['fixtures'][j+1]['exact_stored_determinant']
        m=load();out['extracted_ast_sha256']=m.extracted_ast_sha256
        for arm in p['arms']:
            t=time.process_time();m.set_rules5(arm['thin'],arm['log']);m.THIN_DET5[0]=math.inf if arm['threshold']=='infinity' else arm['threshold']
            out['setup'].append({'arm':arm['name'],'thin_nodes':len(m._DU5[0][0]),'log_nodes':len(m._LOG5[0][0]),'cpu':time.process_time()-t,'scope':'Rule setup timing only; library source/native/setup operations not included in source Ops.'})
            for row in data:
                ops=Ops();record={'fixture':row['name'],'arm':arm['name']};out['rows'].append(record);t=time.process_time();m.STATS['simplices']=0
                try:
                    with warnings.catch_warnings(record=True) as ws:
                        warnings.simplefilter('always');value=float(m._orth5(row['rays'].copy(),ops));record['warnings']=[str(w.message) for w in ws]
                    record['returned_hex']=value.hex();record['finite']=math.isfinite(value)
                    if record['finite']:
                        with mp.workdps(140):
                            vv=mp.mpf(value);ref=mp.mpf(row['reference120']);error=vv-ref
                            record.update(value=value,error=mp.nstr(error,45),absolute_error=float(abs(error)),relative_error=float(abs(error/ref)),absolute_pass=abs(error)<=mp.mpf(str(p['absolute_diagnostic_tolerance'])),relative_pass=abs(error/ref)<=mp.mpf(str(p['relative_diagnostic_tolerance'])),nonnegative=value>=0,probability_range=0<=value<=1)
                except Exception as e:record['exception']=repr(e)
                finally:
                    record.update(cpu=time.process_time()-t,source_proxy_ops=ops.n,source_simplices=m.STATS['simplices']);save()
        assert len(out['rows'])==len(data)*len(p['arms']);out['complete']=True
    except BaseException as e:out['failure']=repr(e);raise
    finally:save();print(json.dumps({k:out[k] for k in ['complete','cpu','max_rss_kib']}))
if __name__=='__main__':main()
