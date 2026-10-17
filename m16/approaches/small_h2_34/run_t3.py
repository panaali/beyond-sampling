import importlib.util,json,time
from pathlib import Path
import numpy as np
from estimator import calculate,forward
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('independent_facet',HERE.parent/'facet_transport'/'probe.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)

def load(T):
    d=json.loads((HERE/'weights_seed0.json').read_text())[f'T{T}']
    return tuple(np.array(d[k]) for k in ('Wi','Wh','Wo'))

def independent(model):
    cells,stages=ref.gate_complex(model);ranked=ref.rank_complex(cells)
    return ref.accuracy(ranked,model[2]),dict(gate_cells=len(cells),rank_cells=len(ranked),partition_mass=sum(ref.area(p) for p,B in cells)/(4*np.pi))

def main():
    start=time.process_time();base=load(3);noise=np.random.default_rng(47001)
    models=[base]+[tuple(w+.05*noise.normal(size=w.shape) for w in base) for _ in range(8)]
    x=np.random.default_rng(47003).normal(size=(65536,3));rows=[]
    for i,m in enumerate(models):
        p,s=calculate(m);q,qs=independent(m)
        timings=[]
        for repeat in range(3):
            begin=time.process_time();forward(m,x);timings.append(time.process_time()-begin)
        per=float(np.median(timings))/len(x);equivalent=s['cpu_seconds']/per
        rows.append(dict(copy=i,accuracy=p,reference=q,difference=p-q,stats=s,reference_stats=qs,
                         baseline_cpu=timings,cpu_equivalent_forwards=equivalent,
                         baseline_rmse_at_equal_cpu=float(np.sqrt(p*(1-p)/max(equivalent,1))),
                         scale_difference=calculate((m[0]*1e-16,m[1],m[2]*1e-16))[0]-p))
    controls=[]
    for label,wo in [('zero',np.zeros((3,2))),('duplicate',np.ones((3,2)))]:
        p,s=calculate((base[0],base[1],wo));controls.append(dict(name=label,accuracy=p,error=p-1/3))
    result=dict(rows=rows,controls=controls,total_cpu=time.process_time()-start,
                benchmark_forward_inputs=9*3*65536,random_inputs_used_by_estimator=0,
                original_checkpoint_arrays_pending_reverification=False,
                classification='analytic tail with floating geometry; independent numeric reference; CPU cost not FLOPs; not self-confirmed')
    (HERE/'results_t3.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cpu=result['total_cpu'],max_diff=max(abs(r['difference']) for r in rows),
        accuracy_range=[min(r['accuracy'] for r in rows),max(r['accuracy'] for r in rows)],
        cpu_fpe_range=[min(r['cpu_equivalent_forwards'] for r in rows),max(r['cpu_equivalent_forwards'] for r in rows)])))
if __name__=='__main__':main()
