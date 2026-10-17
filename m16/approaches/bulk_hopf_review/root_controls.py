"""Predeclared coefficient-zero circle partition controls; preserve source failure first."""
from pathlib import Path
import importlib.util,json,time,hashlib,types
import numpy as np
HERE=Path(__file__).parent
PATH=HERE/'frozen/hopf.py'
SOURCE=PATH.read_text()
assert hashlib.sha256(PATH.read_bytes()).hexdigest()==json.loads((HERE/'manifest.json').read_text())['hopf.py']['sha256']
spec=importlib.util.spec_from_file_location('frozen_hopf',PATH);H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)

def run(module):
    out=[]
    coefficients=np.array([[1.,0.],[0.,1.],[1.,-1.],[0.,0.]])
    for lower in (0.,np.pi):
        lo=np.full(4,lower);hi=np.full(4,2*np.pi)
        ids,roots=module._cuts(lo,hi,coefficients,np.zeros_like(coefficients))
        for row,a in enumerate(coefficients):
            expected=sorted(v for c in a if c!=0 for v in (np.pi/2,3*np.pi/2) if lower<v<2*np.pi)
            got=sorted(roots[ids==row].tolist());same=len(expected)==len(got) and all(abs(x-y)<1e-14 for x,y in zip(expected,got))
            missing=[x for x in expected if not any(abs(x-y)<1e-14 for y in got)]
            witness=None
            if missing:
                root=missing[0];left=max([lower]+[v for v in got if v<root]);right=min([2*np.pi]+[v for v in got if v>root]);p=(left+root)/2;q=(root+right)/2
                assert np.cos(p)*np.cos(q)<0
                witness=dict(first_angle_over_pi=p/np.pi,second_angle_over_pi=q/np.pi,source_interval_over_pi=[left/np.pi,right/np.pi],cosine_signs=[int(np.sign(np.cos(p))),int(np.sign(np.cos(q)))])
            out.append(dict(a=a.tolist(),b=[0.,0.],interval_over_pi=[lower/np.pi,2.],expected_over_pi=[x/np.pi for x in expected],actual_over_pi=[x/np.pi for x in got],root_multiset_matches=same,missing_distinct_roots_over_pi=sorted(set(x/np.pi for x in missing)),classification_witness=witness))
    return out

def main():
    start=time.process_time();original=run(H)
    assert sum(not r['root_multiset_matches'] for r in original)==4
    assert sum(bool(r['classification_witness']) for r in original)==3
    out=dict(original=original,original_failure_preserved=True,scope='Exact coefficient-zero primitive controls, not a saved-trained-panel failure or complete fibre certificate.')
    (HERE/'root_controls.json').write_text(json.dumps(out,indent=2)+'\n')
    # Only after preserving the original result, change the mask ordering alone in an owned-process module.
    old='np.repeat((A != 0) | (B != 0), 2, 1)';new='np.tile((A != 0) | (B != 0), (1, 2))'
    assert SOURCE.count(old)==1
    patched=types.ModuleType('local_tile_rescue');exec(compile(SOURCE.replace(old,new),str(PATH)+' [local tile mask]', 'exec'),patched.__dict__)
    out['tile_rescue']=run(patched);assert all(r['root_multiset_matches'] for r in out['tile_rescue'])
    out['cpu_seconds']=time.process_time()-start
    (HERE/'root_controls.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(original_mismatches=4,original_classification_witnesses=3,rescue_controls=len(out['tile_rescue']),cpu_seconds=out['cpu_seconds'])))
if __name__=='__main__':main()
