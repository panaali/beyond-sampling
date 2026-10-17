"""Frozen complete two-tail comparisons; no random model inputs."""
import hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
from recurrence import SOURCE,baseline,endpoint

def load_tail(name,mass):
    sp=importlib.util.spec_from_file_location(name,SOURCE/'tail.py')
    mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
    mod.tail_mass_with_fallback=mass
    return mod

def main():
    start=time.process_time()
    import torch
    import_cpu=time.process_time()-start
    old=load_tail('old_tail',baseline.tail_mass_with_fallback)
    new=load_tail('endpoint_tail',endpoint.tail_mass_with_fallback)
    rows=[]
    for T in (4,6,8,10):
        path=Path(f'/workspace/clean-neurons/work/checkpoints/H2_T{T}_seed0.pth')
        t=time.process_time();state=torch.load(path,map_location='cpu',weights_only=True)
        model=tuple(state[k].numpy().astype(float).reshape(-1) if k=='rnn.weight_ih_l0' else state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
        loading=time.process_time()-t
        for sign in (1,-1):
            prefix=sign*(-1.)**np.arange(T-2)*np.arange(1,T-1)
            t=time.process_time();h,top=old.prefix_summary(model,prefix);setup=time.process_time()-t
            row={'T':T,'sign':sign,'prefix':prefix.tolist(),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model_loading_cpu':loading if sign==1 else 0.,'prefix_setup_cpu':setup}
            for name,mod in (('old',old),('endpoint',new)):
                t=time.process_time()
                try:
                    p,stats=mod.conditional_probability(model,h,top,T-2)
                    row[name]={'probability':p,'stats':stats,'cold_cpu':time.process_time()-t}
                    t=time.process_time();repeat_values=[]
                    for _ in range(3):
                        v,st=mod.conditional_probability(model,h,top,T-2);repeat_values.append(v)
                    row[name]['repeat_cpu']=time.process_time()-t;row[name]['repeat_values']=repeat_values
                except Exception as e:row[name]={'failure':repr(e),'cpu_seconds':time.process_time()-t}
            if 'probability'in row['old'] and 'probability'in row['endpoint']:
                row['difference']=row['endpoint']['probability']-row['old']['probability']
            rows.append(row)
            if time.process_time()-start>100:raise RuntimeError('CPU guard after saved case required')
    out={'cases':rows,'cpu_seconds':time.process_time()-start,'torch_import_cpu':import_cpu,'tail_source_sha256':hashlib.sha256((SOURCE/'tail.py').read_bytes()).hexdigest(),'moments_source_sha256':hashlib.sha256((SOURCE/'moments.py').read_bytes()).hexdigest(),'sampled_model_inputs':0,'deterministic_prefix_steps':2*sum(T-2 for T in (4,6,8,10)),'portable_cost_certified':False}
    Path(__file__).with_name('pilot_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'cpu_seconds':out['cpu_seconds'],'cases':[{'T':r['T'],'sign':r['sign'],'difference':r.get('difference'),'old':{k:v for k,v in r['old'].items() if k not in ('stats','repeat_values')},'endpoint':{k:v for k,v in r['endpoint'].items() if k not in ('stats','repeat_values')}} for r in rows]},indent=2))
if __name__=='__main__':main()
