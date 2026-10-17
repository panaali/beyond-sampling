import json,time,warnings
from pathlib import Path
warnings.filterwarnings('ignore',message='Unrecognized options detected')
import certify as C
import model as M
P=Path(__file__).resolve().parent;start=time.process_time();out=[]
for L in (((0,)*4,)*4,((1,2,3,4),)*4):
    for target in range(4):
        rows=M.correct_planes(L,target)
        assert (rows is not None)==(target==0)
        for other in range(4):
            if other!=target:assert (M.wrong_planes(L,target,other) is not None)==(other<target)
        out.append(dict(target=target,correct_possible=rows is not None))
planes,L=M.map_and_planes('0'*16,(0,)*4,((0,)*4,)*4,((0,)*4,)*4,1);assert planes==[]
for top in (1,2,3):
    rows=M.rank_planes(top,0)+M.correct_planes(L,0);r=C.feasible(rows);assert r['status']=='feasible' and C.verify(rows,r)
(P/'tie_controls.json').write_text(json.dumps(dict(complete=True,cases=out,zero_gate_forms_removed=True,lp_calls=C.CALLS,cpu_seconds=time.process_time()-start),indent=2)+'\n');print('tie controls pass')
