"""Frozen DD clip fixtures, every-call replay and complete-workload comparison."""
import hashlib,json,resource,time
import numpy as np
from dd_profile import E,ladder,load,HERE,ROOT
import dd_native as N

def same(a,b):
    difference=0.
    for old,new in zip(a,b):
        if old is None or new is None:
            assert old is None and new is None;continue
        assert old[0].shape==new[0].shape and np.array_equal(old[1],new[1])
        difference=max(difference,float(np.max(abs(old[0]-new[0]),initial=0)))
    assert difference<=1e-12
    return difference

def run():
    start=time.process_time();original=E.clip;out={'controls':[],'cases':[]}
    for d in (2,3,4):
        R=np.eye(d);Z=np.zeros((d,2),np.uint64)
        for i in range(d):Z[i,0]=np.uint64(((1<<d)-1)^(1<<i))
        normals=[np.ones(d),-np.ones(d),np.eye(d)[0],np.array([1.,-1.]+[0.]*(d-2))]
        for j,a in enumerate(normals):
            for want in ((True,True),(True,False),(False,True)):
                old=original(R,Z,a,10,E.Ops(),want);new=N.clip(R,Z,a,10,E.Ops(),want)
                out['controls'].append({'dimension':d,'normal':j,'want':want,'difference':same(old,new)})
        old=original(R,Z,np.eye(d)[0],0,E.Ops());new=N.clip(R,Z,np.eye(d)[0],0,E.Ops());same(old,new)
        try:N.clip(R,Z,np.zeros(d),10,E.Ops())
        except ValueError:pass
        else:raise AssertionError('zero normal accepted')
    base,sha=load(ROOT/'work/checkpoints/H16_T3_seed0.pth');out['checkpoint_sha256']=sha
    for sigma in (0.,.0025,.02):
        rng=np.random.default_rng(1);model=base if sigma==0 else tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in base)
        corpus=[];differences=[]
        def checked(R,Z,a,cid,ops,want=(True,True)):
            old=original(R,Z,a,cid,ops,want);new=N.clip(R,Z,a,cid,E.Ops(),want)
            differences.append(same(old,new));corpus.append((R.copy(),Z.copy(),a.copy(),cid,want));return old
        E.clip=checked
        try:expected,info=ladder.exact_acc(model,check=True)
        finally:E.clip=original
        tick=time.process_time();old,oi=ladder.exact_acc(model,check=True);python_cpu=time.process_time()-tick
        E.clip=N.clip
        try:
            tick=time.process_time();new,ni=ladder.exact_acc(model,check=True);native_cpu=time.process_time()-tick
        finally:E.clip=original
        assert abs(new-old)<=1e-12
        for k in ('cells','cells_per_step','wrong_pieces','correct_pieces','simplices','correct_simplices'):assert oi[k]==ni[k],k
        tick=time.process_time();packed=N.pack(corpus);packing=time.process_time()-tick
        wall=time.perf_counter();tick=time.process_time();sums,cost=N.replay(packed);batch=time.process_time()-tick;batch_wall=time.perf_counter()-wall
        wall=time.perf_counter();tick=time.process_time();reference=[]
        for R,Z,a,cid,want in corpus:
            sides=original(R,Z,a,cid,E.Ops(),want);reference.append(sum(float(np.sum(c[0])) for c in sides if c is not None))
        replay=time.process_time()-tick;replay_wall=time.perf_counter()-wall
        residual=float(np.max(abs(sums-reference)));assert residual<1e-12
        row=dict(sigma=sigma,clip_calls=len(corpus),max_ray_difference=max(differences),python_value=old,native_value=new,value_difference=new-old,python_cpu=python_cpu,native_cpu=native_cpu,speed_ratio=python_cpu/native_cpu,python_info=oi,native_info=ni,packing_cpu=packing,batch_cpu=batch,python_replay_cpu=replay,batch_speed_ratio=replay/batch if batch>0 else None,batch_wall=batch_wall,python_replay_wall=replay_wall,batch_wall_speed_ratio=replay_wall/batch_wall,batch_checksum_max_difference=residual,original_clip_operation_counts=cost.tolist())
        out['cases'].append(row);out.update(cpu_seconds=time.process_time()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=hashlib.sha256((HERE/'dd_clip.cpp').read_bytes()).hexdigest())
        (HERE/'dd_results.json').write_text(json.dumps(out,indent=2)+'\n')
        print(sigma,'clip count',len(corpus),'max diff',max(differences),'full speed',row['speed_ratio'],'batch speed',row['batch_speed_ratio'],flush=True)
if __name__=='__main__':run()
