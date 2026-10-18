"""Frozen discrepancy-directed cross-formulation controls; no parameter tuning."""
import json,time,resource
from pathlib import Path
import numpy as np
from ..cone_mass4_review.independent_reference import mp_mass
HERE=Path(__file__).parent

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(110,112));start=time.process_time();rays=np.load(HERE.parent/'class_region_t4/quadratic_exact_mass_calls.npz')['rays'];positive=np.load(HERE/'positive_values.npz');source=np.load(HERE.parent/'cone_mass4_review/quadratic_values.npz')['values'];out=dict(rows=[],complete=False)
    def save():out['cpu']=time.process_time()-start;(HERE/'mp_controls.json').write_text(json.dumps(out,indent=2)+'\n')
    for i in (1459,1458,3021):
        row=dict(index=i,positive_reference=float(positive['values'][i]),integration_only_bound=float(positive['bounds'][i]),author=float(source[i]),precisions=[]);out['rows'].append(row)
        try:
            for dps in (50,70):row['precisions'].append(mp_mass(rays[i],dps,start));save()
            row['positive_minus_mp']=row['positive_reference']-row['precisions'][-1]['value'];row['author_minus_mp']=row['author']-row['precisions'][-1]['value']
        except Exception as exc:row['failure']=type(exc).__name__+': '+str(exc)
        save();print(json.dumps(row),flush=True)
    out['complete']=True;save()
if __name__=='__main__':main()
