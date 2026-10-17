"""Independent high-precision checks of endpoint and angular primitives."""
import json,time
from pathlib import Path
import numpy as np
import mpmath as mp
mp.mp.dps=180
LD=np.longdouble

def exact(x):
    if np.isinf(x):return mp.inf if x>0 else -mp.inf
    a,b=x.as_integer_ratio();return mp.mpf(a)/b

def stable_angle(lo,hi):
    if np.isneginf(lo):return np.arctan2(LD(1),-hi) if hi<=0 else np.arctan(hi)+np.arccos(LD(-1))/2
    if np.isposinf(hi):return np.arctan2(LD(1),lo) if lo>=0 else np.arccos(LD(-1))/2-np.arctan(lo)
    if lo<0 and hi>0:return np.arctan(hi)-np.arctan(lo)
    x,y=(lo,hi) if lo>=0 else (-hi,-lo)
    scale=max(LD(1),y)
    return np.arctan2((y-x)/scale,1/scale+x*(y/scale))

def run():
    start=time.process_time();rows=[]
    for name,lo,hi in [('adjacent_one',LD(1),np.nextafter(LD(1),LD(2))),('large_narrow',LD('1e12'),LD('1e12')+1),('extreme_adjacent',LD('1e100'),np.nextafter(LD('1e100'),LD('inf'))),('negative_large',-LD('1e12')-1,-LD('1e12')),('crossing',LD(-1),LD(2)),('positive_tail',LD('1e100'),LD('inf')),('negative_tail',-LD('inf'),-LD('1e100')),('whole_line',-LD('inf'),LD('inf'))]:
        truth=mp.atan(exact(hi))-mp.atan(exact(lo));old=np.arctan(hi)-np.arctan(lo);new=stable_angle(lo,hi)
        row=dict(name=name,lo=str(lo),hi=str(hi),truth=str(truth),old=str(old),stable=str(new),old_relative_error=float(abs(exact(old)-truth)/truth),stable_relative_error=float(abs(exact(new)-truth)/truth))
        rows.append(row);assert row['stable_relative_error']<1e-17
    lo=LD(1);hi=np.nextafter(lo,LD(2));mid=LD('.5')*lo+LD('.5')*hi
    # a+b*t=t-lo is strictly positive on this entire interval.
    midpoint=bool(mid-lo>0);right_limit=bool(lo>=lo)
    assert not midpoint and right_limit
    out=dict(angle_controls=rows,adjacent_classification=dict(lo=str(lo),hi=str(hi),midpoint=str(mid),old_positive=midpoint,right_limit_positive=right_limit),cpu_seconds=time.process_time()-start)
    (Path(__file__).parent/'endpoint_controls.json').write_text(json.dumps(out,indent=2)+'\n')
    print([(r['name'],r['old_relative_error'],r['stable_relative_error']) for r in rows])
if __name__=='__main__':run()
