"""Read-only complete T46 geometry profiling; no shared mutations."""
import cProfile,hashlib,importlib.util,json,pstats,resource,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
from benchmark import load,HERE,ROOT
DD=ROOT/'m16/approaches/exact_ladder'
sys.path.insert(0,str(DD))
import cones as E
import ladder

def run():
    start=time.process_time();base,sha=load(ROOT/'work/checkpoints/H16_T3_seed0.pth')
    out={'checkpoint_sha256':sha,'source_sha256':{f:hashlib.sha256((DD/f).read_bytes()).hexdigest() for f in ('cones.py','ladder.py')},'cases':[]}
    for sigma in (0.,.0025,.02):
        rng=np.random.default_rng(1);model=base if sigma==0 else tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in base)
        tick=time.process_time();value,info=ladder.exact_acc(model,check=True);cpu=time.process_time()-tick
        original=E.clip;shapes=Counter();calls=[0]
        def recorded(R,Z,a,cid,ops,want=(True,True)):
            shapes[f'{len(R)}x{R.shape[1]}']+=1;calls[0]+=1
            return original(R,Z,a,cid,ops,want)
        E.clip=recorded;prof=cProfile.Profile();prof.enable()
        try:again,info2=ladder.exact_acc(model,check=True)
        finally:prof.disable();E.clip=original
        assert value==again and info['cells']==info2['cells'] and info['simplices']==info2['simplices']
        stats=pstats.Stats(prof);rows=[]
        for (file,line,name),(primitive,total,selftime,cumulative,callers) in stats.stats.items():rows.append(dict(file=file,line=line,name=name,primitive_calls=primitive,calls=total,self_seconds=selftime,cumulative_seconds=cumulative))
        row=dict(sigma=sigma,value=value,repeat_difference=again-value,unprofiled_cpu=cpu,info=info,clip_calls=calls[0],clip_shapes=dict(shapes),profile_total=stats.total_tt,top_cumulative=sorted(rows,key=lambda r:-r['cumulative_seconds'])[:25],top_self=sorted(rows,key=lambda r:-r['self_seconds'])[:25])
        out['cases'].append(row);out.update(cpu_seconds=time.process_time()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (HERE/'dd_profile.json').write_text(json.dumps(out,indent=2)+'\n')
        print(sigma,value,cpu,'clipcalls',calls[0],[(r['name'],round(r['cumulative_seconds'],4)) for r in row['top_cumulative'][:8]],flush=True)
if __name__=='__main__':run()
