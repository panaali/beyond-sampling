"""Five frozen synthetic source paths: bit parity and a partial call census."""
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
import warnings
import numpy as np
from loader import load,Ops
from trace import Trace

HERE=Path(__file__).resolve().parent
OUT=HERE/'pilot.json'

def main():
    assert not OUT.exists(),'Preserve previous pilot'
    start=time.process_time();gate=json.loads((HERE/'pilot_gate.json').read_text())
    for name,sha in gate['source_sha256'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha,name
    controls=json.loads((HERE/'controls.json').read_text())
    assert controls['complete'] and len(controls['rows'])==8
    for name,sha in controls['source_sha256'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha,name
    saved=json.loads((HERE/'wedge_results.json').read_text());assert saved['complete']
    fixtures={r['name']:r for r in saved['fixtures']}
    source_rows={r['fixture']:r for r in saved['rows'] if r['arm']=='source'}
    expected=json.loads((HERE/'expected_counts.json').read_text())
    normal,_=load(False);adapted,_=load(True)
    out={'complete':False,'dependency_gate':gate,'rows':[],'setup':[],
         'python_version':sys.version,'numpy_version':np.__version__,
         'pilot_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'scope':'Five pre-existing synthetic fixtures, two executions each for source/instrumented parity. No trained corpus, model forwards or MC. Element counts and opaque calls are not full native prices.'}
    try:
        for name,m in [('baseline',normal),('adapted',adapted)]:
            t=time.process_time();m.set_rules5(6,128)
            out['setup'].append({'arm':name,'cpu_seconds':time.process_time()-t,'thin_nodes':len(m._DU5[0][0]),'log_nodes':len(m._LOG5[0][0]),'numeric_cost':'Unpriced model-free setup; executed, not inferred free runtime.'})
        for name in gate['fixtures']:
            rays=np.array([[float.fromhex(x) for x in row] for row in fixtures[name]['rays_hex']],dtype=np.float64)
            before=hashlib.sha256(rays.tobytes()).hexdigest()
            tr=Trace();adapted.__dict__['__trace']=tr
            row={'fixture':name,'rays_sha256':before,'saved_source_record':source_rows[name],'arms':{}}
            out['rows'].append(row)
            for label,m in [('baseline',normal),('adapted',adapted)]:
                m.STATS['simplices']=0;ops=Ops();t=time.process_time()
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter('always');value=float(m._orth5(rays,ops))
                row['arms'][label]={'returned_hex':value.hex(),'value':value,'proxy_ops':ops.n,'simplices':m.STATS['simplices'],'cpu_seconds':time.process_time()-t,'warnings':[str(w.message) for w in ws]}
                assert hashlib.sha256(rays.tobytes()).hexdigest()==before
                assert math.isfinite(value)
            row['trace']=tr.serial()
            row['branch']='thin' if name.endswith('22') else 'log'
            b=row['arms']['baseline'];a=row['arms']['adapted'];s=source_rows[name]
            assert b['returned_hex']==a['returned_hex']==s['returned_hex'],name
            assert b['proxy_ops']==a['proxy_ops']==s['source_proxy_ops']==(51965 if row['branch']=='thin' else 173060)
            assert b['simplices']==a['simplices']==s['source_simplices']==1
            assert a['warnings']==b['warnings']==s['warnings']
            assert dict(tr.counts)==expected[row['branch']],(name,dict(tr.counts),expected[row['branch']])
            row['parity_and_hand_count_pass']=True
        out['complete']=True
    except BaseException as e:out['failure']=repr(e);raise
    finally:
        out.update(cpu_seconds=time.process_time()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   synthetic_mass_calls=sum(len(r['arms']) for r in out['rows']),trained_model_MC_calls=0)
        OUT.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'complete':out['complete'],'synthetic_mass_calls':out['synthetic_mass_calls'],
                     'cpu_seconds':out['cpu_seconds'],'rows':[{k:r[k] for k in ['fixture','branch','parity_and_hand_count_pass']} for r in out['rows']]}))

if __name__=='__main__':main()
