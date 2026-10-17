"""Fixed-panel numerical truth, separate from estimator work and cost."""
from pathlib import Path
import importlib.util,json,time
from models import frozen_models
HERE=Path(__file__).parent

def main():
    p=HERE/'snapshot/independent_t3_reference.py';spec=importlib.util.spec_from_file_location('independent_t3',p);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    panel=json.loads((HERE/'panel_results.json').read_text());lookup={r['name']:r for r in panel['rows']};rows=[];started=time.process_time()
    for name,model,digest in frozen_models():
        stamp=time.process_time();truth,leaves=ref.exact(model);r=lookup[name];estimate=r['baseline']['value'];error=estimate-truth;budget=r['expression_sample_equivalents'];variance=truth*(1-truth)/budget
        if name in ('zero','constant_zero_readout','constant_duplicate_readout'):assert abs(truth-1/3)<1e-14
        out=dict(name=name,truth=truth,truth_leaves=leaves,estimate=estimate,error=error,expression_sample_equivalents=budget,bernoulli_variance=variance,conditional_squared_bias_ratio=error*error/variance,reference_cpu_seconds=time.process_time()-stamp);rows.append(out);print(json.dumps(out),flush=True)
    result=dict(rows=rows,cpu_seconds=time.process_time()-started,scope='independent numerical spherical polygon validation on fixed12; conditional source-expression comparison, not full ARC protocol or certified numerical enclosure')
    (HERE/'reference_results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
