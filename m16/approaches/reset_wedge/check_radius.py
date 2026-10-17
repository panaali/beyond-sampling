import json,time
from pathlib import Path
from .two_suffix import integrate

def main():
    start=time.process_time();path=Path(__file__).parent;m=json.loads((path/'results.json').read_text())['models'][0];wi,wh,wo=m['weights'];runs=[]
    for n in (8,16,32):
        v,s=integrate(wi,wh,wo,n,radial_rule='legendre');runs.append(dict(estimate=v,stats=s))
    if abs(runs[-1]['estimate']-runs[-2]['estimate'])>1e-10:
        v,s=integrate(wi,wh,wo,64,radial_rule='legendre');runs.append(dict(estimate=v,stats=s))
    old=json.loads((path/'terminal_results.json').read_text())['models'][0]
    out=dict(complete=True,model_sha256=m['model_sha256'],runs=runs,partial_three_strata=old['partial_accuracy']+runs[-1]['estimate'],unresolved_event_mass=old['unresolved_event_mass']-runs[-1]['stats']['event_mass_analytic'],cpu_seconds=time.process_time()-start,random_model_inputs=0,formal_float_enclosure=False)
    assert out['cpu_seconds']<18
    (path/'radius_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
