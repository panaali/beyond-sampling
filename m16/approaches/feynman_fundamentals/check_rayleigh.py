"""Closed-form k=2 constant-readout checks; no numerical reference integration."""
import json
from pathlib import Path
import time
import numpy as np
from tail import conditional_probability,prefix_summary

start=time.process_time()
rows=[]
model=(np.ones(1),np.zeros((1,1)),np.zeros((4,1)))
for prefix in ([2.,1.],[1.,2.],[-1.,-2.],[-2.,-1.]):
    h,top=prefix_summary(model,np.array(prefix))
    p,stats=conditional_probability(model,h,top,2)
    u=prefix[0]/np.linalg.norm(prefix)
    c=u/np.sqrt(1+u*u)
    # R~chi_2: E Phi(uR)=1/2+c/2; integration by parts gives its square.
    mean=.5+.5*c
    square=.25+.5*c+c*np.arctan(c)/np.pi
    expected=2*(mean-square) if prefix[0]>prefix[1] else square
    rows.append(dict(prefix=prefix,probability=p,closed_form=expected,error=p-expected,stats=stats))
output=dict(cases=rows,cpu_seconds=time.process_time()-start,full_model_inputs=0,
            deterministic_prefix_sequences=4,deterministic_prefix_steps=8)
Path(__file__).with_name("rayleigh_check.json").write_text(json.dumps(output,indent=2)+"\n")
print(json.dumps(output))
