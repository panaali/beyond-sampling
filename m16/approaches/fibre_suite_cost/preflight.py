"""All471 identities and analytic constant controls before native suite recount."""
import time
START=time.process_time()
import json,resource
import numpy as np
from .common import HERE,verify_sources,models,references,digest,evaluate,sample_count

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,20));out=dict(source_equivalence=verify_sources(),models=[],controls=[],complete=False)
    for h in (4,5,6):
        panel=models(h);old,prior=references(h);assert list(panel)==list(old)
        for name,model in panel.items():
            sha=digest(model);assert sha==prior[name]['weight_sha256'],(h,name);assert prior[name]['truth']==old[name]['truth'];out['models'].append(dict(H=h,name=name,sha256=sha))
        for name,model in [('zero',(np.zeros(h),np.zeros((h,h)),np.zeros((3,h)))),('constant_zero_readout',(np.arange(1,h+1,dtype=float),np.eye(h),np.zeros((3,h)))),('constant_duplicate_readout',(np.arange(1,h+1,dtype=float),np.eye(h),np.ones((3,h))))]:
            r=evaluate(model);r.update(H=h,name=name,expected=1/3);assert abs(r['value']-1/3)<1e-14;sample=sample_count(model);den=sample['named_numeric_total']+sample['integer_control_total'];assert den==12*h+4*h*h;r['sampler_total']=den;out['controls'].append(r)
    out.update(complete=True,cpu_including_imports=time.process_time()-START);(HERE/'preflight.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(complete=True,models=len(out['models']),controls=len(out['controls']),sampler_totals=sorted(set(x['sampler_total'] for x in out['controls'])),cpu=out['cpu_including_imports']),indent=2))
if __name__=='__main__':main()
