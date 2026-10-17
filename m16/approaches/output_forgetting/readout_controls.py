from pathlib import Path
from fractions import Fraction as F
import json,time
from readout_cone import certificate

def main():
    start=time.process_time();rows=[]
    readouts={'axis':[[1,0],[0,1]],'zero':[[0,0]]*3,'duplicate':[[1,0],[1,0],[0,1],[0,1]],'intermediate':[[1,0],[0,1],[F(9,10),F(9,10)]]}
    for name,wo in readouts.items():
        for sign in ('+','-'):
            r=certificate(F(1),F(1),F(1,4),F(1),wo,sign);r['control']=name;rows.append(r)
    out={'cases':rows,'cpu_seconds':time.process_time()-start,'exact_arithmetic':True}
    Path(__file__).with_name('readout_controls_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'cpu_seconds':out['cpu_seconds'],'thresholds':[(r['control'],r['sign'],r['old_threshold'],r['threshold']) for r in rows]},indent=2))
if __name__=='__main__':main()
