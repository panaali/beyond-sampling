"""Cost-only plan artifact plus independent exact-line controls."""
import time
START=time.process_time()
import hashlib,importlib.util,json,resource,sys
from pathlib import Path
import numpy as np
import engine as E

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'walker_tolerance_review'))
from check import fixtures
from rational_line import wrong_mass


def main():
    controls=[]
    for name,M,y,a,b in fixtures():
        d=np.zeros(len(y));d[a]=1/np.sqrt(2);d[b]=-1/np.sqrt(2)
        reference=wrong_mass(M,y,d,-.25,.25,b,a,deadline=START+18)
        mass,cost,cells=E.C.walk_mass(M,y[None],np.array([a]),np.array([b]),np.array([.25]),'23')
        error=float(mass[0]-reference['mass']);assert abs(error)<1e-12
        controls.append(dict(name=name,error=error,walk_fpe=float(cost[0]),cells=int(cells[0])))
    rows=[];comparisons=[];table_cpu={}
    for T in [4,5,6]:
        begin=time.process_time();E.C.tables(T);table_cpu[str(T)]=time.process_time()-begin
        for seed in range(5):
            M=E.model(T,seed);plan=E.calibrate(M)
            rows.append(dict(H=16,T=T,seed=seed,checkpoint_sha256=E.MODELS[16,T,seed]['sha256'],**plan))
            if seed==0:
                for kind in ['23','12']:
                    y,a,b,tau=E.C.face_points(T,kind,E.C.sobol(T,16,44031))
                    current,cost,cells=E.C.walk_mass(M,y,a,b,tau,kind)
                    old,oldcost,oldcells=E.C.geometry.walk_mass(M,y,a,b,tau,kind)
                    comparisons.append(dict(T=T,kind=kind,max_absolute_difference=float(np.max(abs(current-old))),
                        current_walk_fpe=float(cost.sum()),legacy_walk_fpe=float(oldcost.sum()),
                        current_cells=int(cells.sum()),legacy_cells=int(oldcells.sum())))
            print(T,seed,[(p['budget'],p['faces_per_kind']) for p in plan['plans']],flush=True)
            if time.process_time()-START>18:raise TimeoutError('review CPU cap')
    out=dict(models=rows,controls=controls,trained_common_faces=comparisons,table_cpu_seconds=table_cpu,
        source_pins=E.PINS,engine_sha256=hashlib.sha256((HERE/'engine.py').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        note='Calibration costs only; no production estimates or truth table read. Plans frozen before outcomes.')
    (HERE/'calibration.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['models','controls']},indent=2))


if __name__=='__main__':main()
