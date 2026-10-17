"""Frozen exact-angle rank-degenerate Hopf limit; no input grid or random inputs."""
from pathlib import Path
import importlib.util,json,math,time
import numpy as np
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('pinned_hopf',HERE/'frozen/hopf.py');H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)

def main():
    start=time.process_time();rows=[]
    model=(np.zeros(1),np.zeros((1,1)),np.zeros((4,1)))
    ratios=[.5,2.]+[1+sign*2.**(-k) for k in (2,4,8,16) for sign in (-1,1)]
    for r in ratios:
        norm=math.sqrt(1+r*r);a=np.array([r,0.,0.,1.])/norm;b=np.array([0.,r,-1.,0.])/norm
        assert abs(float(a@a)-1)<5e-16 and abs(float(b@b)-1)<5e-16 and float(a@b)==0
        expected=.5 if r<1 else (3*math.pi/4-math.atan(r))/(2*math.pi)
        value,pieces=H.fibre(model,a,b)
        assert abs(value-expected)<2e-15,(r,value,expected)
        rows.append(dict(r=r,analytic_fraction=expected,source_fraction=value,difference=value-expected,source_pieces=pieces))
    out=dict(rows=rows,left_limit=.5,right_limit=.25,model='H1T4 all-zero weights; first-index output always0',scope='Universal fibre-continuity counterexample at rank-degenerate circle. No trained convergence/rate failure, no whole-estimator cost or global quadrature certificate.',cpu_seconds=time.process_time()-start,source_circle_integrals=len(rows),random_inputs=0,dense_angular_grid=0,checkpoint_loads=0)
    (HERE/'continuity_control.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(cases=len(rows),max_difference=max(abs(r['difference']) for r in rows),cpu_seconds=out['cpu_seconds'])))
if __name__=='__main__':main()
