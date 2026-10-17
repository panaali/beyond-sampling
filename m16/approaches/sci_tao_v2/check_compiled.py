"""Bounded reference parity and CPU benchmarks; no additional ground-truth draws."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import core as C
import compiled as K
from check import checkpoint_model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    args=parser.parse_args()
    rng=np.random.default_rng(29341)
    models=[]
    for H,T in ((1,2),(2,3),(4,3),(5,5),(8,6),(12,8),(16,10),(32,5)):
        models.append((f'random_dev_H{H}_T{T}',(rng.normal(size=H),rng.normal(size=(H,H))/np.sqrt(H),rng.normal(size=(T,H))),64))
    models.append(('all_zero',(np.zeros(3),np.zeros((3,3)),np.zeros((4,3))),64))
    base=checkpoint_model(args.checkpoint)
    for sigma in (.0025,.02):
        dev_rng=np.random.default_rng(1)
        m=tuple(w+sigma*np.sqrt((w*w).mean())*dev_rng.standard_normal(w.shape) for w in base)
        models.append((f'dev_s{sigma}_0',m,256))
    rows=[]
    for name,model,n in models:
        T=model[2].shape[0]
        row={'model':name,'lines':0,'max_mass_difference':0.,'python_seconds':0.,'cpp_seconds':0.}
        for kind in (('12',) if T==2 else ('12','23')):
            ys=C.face_points(T,kind,C.sobol(T,n,734))
            args=(model,*ys,kind)
            start=time.process_time();ref=C.walk_mass(*args,details=True);row['python_seconds']+=time.process_time()-start
            start=time.process_time();new=K.walk_mass(*args,details=True);row['cpp_seconds']+=time.process_time()-start
            delta=float(np.max(np.abs(new[0]-ref[0])))
            assert delta<1e-10,(name,kind,delta)
            assert np.array_equal(new[2],ref[2]),(name,kind,'cells')
            for key in ref[3]:
                assert np.array_equal(new[3][key],ref[3][key]),(name,kind,key)
            assert np.array_equal(new[1],ref[1]),(name,kind,'cost')
            row['lines']+=n
            row['max_mass_difference']=max(row['max_mass_difference'],delta)
        row['speedup']=row['python_seconds']/row['cpp_seconds']
        rows.append(row)
    tiny=(np.ones(1),np.zeros((1,1)),np.array([[0.],[1e-14],[0.]]))
    args=(tiny,np.array([[0.,0.,1.]]),np.array([0]),np.array([1]),np.array([.25]),'23')
    assert np.allclose(K.walk_mass(*args)[0],C.walk_mass(*args)[0],atol=1e-15,rtol=0.)
    zero=(*args[:4],np.zeros(1),args[-1])
    assert all(np.all(v==0) for v in K.walk_mass(*zero))
    try:
        K.walk_mass(*args,max_cells=0)
        raise AssertionError('Expected cell-limit exception')
    except RuntimeError:
        pass
    # Warm 4096-line trained DEV batch measures amortized overhead. No tuning.
    perf=[]
    for name,model,_ in models[-2:]:
        ys=C.face_points(10,'23',C.sobol(10,4096,734))
        args=(model,*ys,'23')
        K.walk_mass(*args)
        timing={}
        outputs={}
        for key,fn in (('old',C.geometry.walk_mass),('python',C.walk_mass),('cpp',K.walk_mass)):
            samples=[]
            for _ in range(3):
                start=time.process_time();outputs[key]=fn(*args);samples.append(time.process_time()-start)
            timing[key]=float(np.median(samples))
        assert np.max(np.abs(outputs['cpp'][0]-outputs['python'][0]))<1e-10
        perf.append({'model':name,'lines':4096,'cpu_seconds_median3':timing,
                     'speedup_vs_python':timing['python']/timing['cpp'],
                     'speedup_vs_old':timing['old']/timing['cpp'],
                     'cost_fpe':float(outputs['cpp'][1].sum())})
    # Six frozen DEV estimator calls, comparing two backends; no fresh truths.
    import estimator as E
    estimates=[]
    py_fn=E.C.walk_mass
    for name,model,_ in models[-2:]:
        for budget in (1000,4000,16000):
            E.C.walk_mass=py_fn
            start=time.process_time();ref=E.est(model,budget);pt=time.process_time()-start
            E.C.walk_mass=K.walk_mass
            start=time.process_time();new=E.est(model,budget);ct=time.process_time()-start
            assert abs(ref[0]-new[0])<1e-10 and ref[1]==new[1],(name,budget,ref,new)
            estimates.append({'model':name,'budget':budget,'estimate':new[0],'cost_fpe':new[1],
                              'python_cpu_seconds':pt,'cpp_cpu_seconds':ct,'speedup':pt/ct})
    E.C.walk_mass=py_fn
    out={'parity':rows,'guards':'all-zero, tiny-gap, zero-width, cap failure passed','warm_timing':perf,'estimator_parity':estimates}
    Path(__file__).with_name('compiled_validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
