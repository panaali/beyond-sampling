"""Analytic reset controls and fixed trained geometry parity, unchanged T71 mass rule."""
import time
START=time.process_time()
import hashlib,json,resource,sys
from pathlib import Path
import numpy as np
import check as C
P=Path(__file__).resolve().parent
T71=Path('/workspace/work/recover-small_h2_56/m16/approaches/small_h2_56/odd_tetra');sys.path.insert(0,str(T71))
from counted import Audit,volume

def main():
    control=json.loads((P/'controls.json').read_text());rows=[]
    for r in control['cases']:
        if r['analytic_accuracy'] is None:continue
        model=tuple(np.array(w) for w in r['weights']);g=C.NEW.geometry(model);audit=Audit();tick=time.process_time()
        value,info=volume(np.swapaxes(g['good'],1,2).copy(),audit,order=24,inner_budget=1e-9)
        error=value-r['analytic_accuracy'];assert abs(error)<1e-8
        rows.append(dict(name=r['name'],value=value,truth=r['analytic_accuracy'],difference=error,integration=info,mass_only_cost=audit.report(),cpu_seconds=time.process_time()-tick))
    base,sha=C.REF.load(Path('/workspace/clean-neurons/work/checkpoints/H2_T5_seed0.pth'));trained=[]
    for sigma in [0.,.0025,.02]:
        rng=np.random.default_rng(1);model=base if sigma==0 else tuple(w+rng.normal(size=w.shape)*np.sqrt(np.mean(w*w))*sigma for w in base)
        tick=time.process_time();a=C.OLD.geometry(model);oldcpu=time.process_time()-tick;tick=time.process_time();b=C.NEW.geometry(model);newcpu=time.process_time()-tick
        equality={key:bool(np.array_equal(a[key],b[key])) for key in ['good','bad','rays','bits','headers','maps']};assert all(equality.values())
        assert np.array_equal(a['counts'],b['counts'][:13]) and b['counts'][13]==0
        trained.append(dict(sigma=sigma,model_sha256=hashlib.sha256(b''.join(w.tobytes() for w in model)).hexdigest(),arrays_bit_identical=equality,
            old_cpu=oldcpu,new_cpu=newcpu,counts=b['counts'].tolist()))
    result=dict(complete=True,analytic_controls=rows,trained=trained,checkpoint_sha256=sha,
        source_pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P/'validate.py',T71/'counted.py',T71/'fixed_tangent.py',T71/'primitive.py']},
        cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    assert result['cpu_seconds_including_imports']<95
    (P/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['analytic_controls','source_pins']},indent=2))
if __name__=='__main__':main()
