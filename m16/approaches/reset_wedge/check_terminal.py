import json,time
from pathlib import Path
from .terminal import calculate,ray_label

def main():
    start=time.process_time();path=Path(__file__).parent;data=json.loads((path/'results.json').read_text());rows=[]
    assert ray_label([1.,1.],[[0.,0.]]*8)==0
    for m in data['models']:
        wi,wh,wo=m['weights'];r=calculate(wi,wo,m['kappa_float']);r['sigma']=m['sigma'];r['model_sha256']=m['model_sha256']
        assert abs(r['last_T']['estimate']-r['last_T']['independent_identity'])<2e-10
        if 'estimate' in r['last_T_minus_1']:assert abs(r['last_T_minus_1']['estimate']-r['last_T_minus_1']['independent_reversed_beta'])<2e-10
        rows.append(r)
    out=dict(complete=True,models=rows,cpu_seconds=time.process_time()-start,random_model_inputs=0)
    assert out['cpu_seconds']<18
    (path/'terminal_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
