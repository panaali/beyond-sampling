"""One predeclared complete-circle analytic counterexample; original then tile-only rescue."""
from pathlib import Path
import importlib.util,types,json,time,math
import numpy as np
HERE=Path(__file__).parent;PATH=HERE/'frozen/hopf.py'
spec=importlib.util.spec_from_file_location('pinned_hopf',PATH);H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)

def main():
    start=time.process_time();wi=np.array([1.,0.]);wh=np.zeros((2,2));wo=np.zeros((4,2));wo[1,0]=1.
    model=(wi,wh,wo);a=np.array([2.,0.,0.,1.])/math.sqrt(5);b=np.array([0.,2.,-1.,0.])/math.sqrt(5)
    exact=(math.pi/4-math.atan(.5))/(2*math.pi)
    original,n=H.fibre(model,a,b)
    out=dict(wi=wi.tolist(),wh=wh.tolist(),wo=wo.tolist(),a=a.tolist(),b=b.tolist(),analytic_fraction=exact,original_fraction=original,original_difference=original-exact,original_pieces=n,scope='Explicit H2T4 synthetic whole-circle counterexample, not a trained-panel or complete Hopf-grid failure.')
    path=HERE/'full_circle_control.json';path.write_text(json.dumps(out,indent=2)+'\n')
    src=PATH.read_text();old='np.repeat((A != 0) | (B != 0), 2, 1)';assert src.count(old)==1
    mod=types.ModuleType('tile_rescue');exec(compile(src.replace(old,'np.tile((A != 0) | (B != 0), (1, 2))'),str(PATH)+' [local tile rescue]','exec'),mod.__dict__)
    corrected,n=mod.fibre(model,a,b);out.update(tile_rescue_fraction=corrected,tile_rescue_difference=corrected-exact,tile_rescue_pieces=n,cpu_seconds=time.process_time()-start)
    path.write_text(json.dumps(out,indent=2)+'\n')
    assert abs(original-exact)>.01
    assert abs(corrected-exact)<2e-15
    print(json.dumps(out))
if __name__=='__main__':main()
