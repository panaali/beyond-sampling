"""Same frozen64 cones, original grouped versus guarded grouped selection."""
from pathlib import Path
import hashlib,importlib.util,json,statistics,time
import numpy as np
from radial_probe import SOURCE,Audit
from selector import Guarded
assert hashlib.sha256((SOURCE/'pilot.py').read_bytes()).hexdigest()=='e5878c744621471ed2ccba2c530293982ddef31571cf93ea03f7e551cf8481af'
spec=importlib.util.spec_from_file_location('original_grouped_pilot',SOURCE/'pilot.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
HERE=Path(__file__).resolve().parent


def main():
    start=time.process_time()
    expected={'projected_correct_rays.npz':'15ff556d1ef6168275d67a785eb7f2623a411cb8832816cf5cfc97060cd339c7',
              'plan_1e-06.npz':'3126342f2705bf7db13483b603f423c0d9a036a02eb434b5628913850013ae81'}
    for name,sha in expected.items():assert hashlib.sha256((original.CACHE/name).read_bytes()).hexdigest()==sha
    rays=np.load(original.CACHE/'projected_correct_rays.npz')['rays']
    orders=np.load(original.CACHE/'plan_1e-06.npz')['orders'];indices=np.linspace(0,len(rays)-1,64,dtype=int)
    def run(guarded):
        cpu=time.process_time();audit=Audit();engine=original.Mass(audit,True)
        if guarded:engine.triangle=Guarded(audit)
        rows=[]
        for index in indices:
            value,error,series,stats=engine.one(rays[index],int(orders[index]),1e-8/len(rays))
            rows.append(dict(index=int(index),order=int(orders[index]),mass=value,inner_bound=error,series_bound=series,stats=stats))
        result=dict(cpu=time.process_time()-cpu,rows=rows,cost=audit.report(),
                    partial_mass=sum(r['mass'] for r in rows),selection_cpu=engine.triangle.selection_cpu,
                    node_cpu=engine.triangle.node_cpu,groups=engine.triangle.groups)
        if guarded:result.update(scalar_rescues=engine.triangle.scalar_rescues,
                                 scalar_checks=engine.triangle.scalar_checks,vector_trials=engine.triangle.vector_trials)
        return result
    baseline=run(False);candidate=run(True)
    assert baseline['rows']==candidate['rows']
    timings=[]
    for i in range(10):
        guarded=bool(i%2);result=run(guarded)
        assert result['rows']==(candidate if guarded else baseline)['rows']
        timings.append(dict(variant='guarded' if guarded else 'grouped',cpu=result['cpu'],
                            selection_cpu=result['selection_cpu'],node_cpu=result['node_cpu']))
    medians={name:statistics.median(r['cpu'] for r in timings if r['variant']==name) for name in ('grouped','guarded')}
    out=dict(source_hashes=expected,indices=indices.tolist(),baseline=baseline,candidate=candidate,
             timings=timings,median_cpu=medians,guarded_over_grouped=medians['guarded']/medians['grouped'],
             full_accuracy_estimated=False,portable_cost_gain=False,complete_calls=12,cpu_seconds=time.process_time()-start)
    (HERE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('median_cpu','guarded_over_grouped','complete_calls','cpu_seconds')}))


if __name__=='__main__':main()
