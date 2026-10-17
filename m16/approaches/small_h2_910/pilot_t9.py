"""T54 bounded local pilot, using T53's shared training implementation."""
import hashlib
import importlib.util
import json
import pathlib
import sys
import time
import numpy as np
from check_trained_tail import execute,indicators

HERE=pathlib.Path(__file__).resolve().parent
ENGINE=HERE.parent/'small_h2_78/train_pilot.py'
if not ENGINE.exists():
    ENGINE=pathlib.Path('/workspace/work/recover-small_h2_78/m16/approaches/small_h2_78/train_pilot.py')
spec=importlib.util.spec_from_file_location('shared_local_pilot',ENGINE)
engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
FUND=HERE.parent/'feynman_fundamentals'
sys.path.insert(0,str(FUND))
from tail import conditional_probability,prefix_summary


def main():
    weights,provenance=engine.train(seq_len=9,seed=0,max_steps=2000,max_cpu=30,heldout_seed=54003,out_dir=HERE)
    provenance['training_engine_sha256']=hashlib.sha256(ENGINE.read_bytes()).hexdigest()
    (HERE/'LOCALPILOT_H2_T9_seed0.json').write_text(json.dumps(provenance,indent=2)+'\n')
    model=tuple(weights[key].astype(float) for key in ('Wi','Wh','Wo'))
    prefixes=np.random.default_rng(54010).standard_normal((8,7))
    rows=[];clock=time.process_time()
    for i,prefix in enumerate(prefixes):
        h,top=prefix_summary(model,prefix)
        try:
            p,stats=conditional_probability(model,h,top,7)
            row={'prefix':i,'probability':p,'stats':stats}
            if i<2:
                rng=np.random.default_rng(54020+i)
                x=np.column_stack((prefix[None,:]/np.linalg.norm(prefix)*np.sqrt(rng.chisquare(7,size=32768))[:,None],rng.standard_normal((32768,2))))
                y=indicators(model,x);mc=float(y.mean());se=float(np.sqrt(mc*(1-mc)/len(y)))
                row['validation']={'inputs':len(y),'estimate':mc,'se':se,'z':(p-mc)/se if se else None}
            rows.append(row)
        except ArithmeticError as error:
            rows.append({'prefix':i,'failure':str(error)})
    output={'provenance':'LOCALPILOT, not a published Zoo checkpoint','cases':rows,
        'conditional_validation_inputs':sum(r.get('validation',{}).get('inputs',0) for r in rows),
        'random_outer_prefixes':8,'prefix_recurrent_steps':56,
        'cpu_seconds':time.process_time()-clock,'failures':sum('failure' in r for r in rows)}
    if not output['failures']:
        values=[r['probability'] for r in rows]
        output.update(outer_estimate=float(np.mean(values)),outer_se=float(np.std(values,ddof=1)/np.sqrt(8)))
    (HERE/'pilot_tail_results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))

if __name__=='__main__':main()
