"""T7 application of T50 rank-conditioned transport with stage-dependent boxes.

Numerical collocation, not certified probability integration. Imported algorithm
sources are hashed in the result. No random model-input evaluations are used.
"""
from collections import defaultdict
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from envelope import nonvanishing_certificate

HERE=Path(__file__).resolve().parent
TRANSPORT=Path(os.environ.get('SMALL_TRANSPORT_DIR',str(HERE.parent/'small_h2_910')))
if not (TRANSPORT/'doob.py').exists():
    TRANSPORT=Path('/workspace/work/recover-small_h2_910/m16/approaches/small_h2_910')
sys.path.insert(0,str(TRANSPORT))
from doob import RankRules,conditional_rules
from backward import target_kernel_grid,FUND


def stage_grids(model,j,n):
    wi,wh,wo=model;T=len(wo);npre=T-2 if j<T-2 else T-3
    B=np.zeros(2);boxes=[B.copy()];grids=[]
    def grid(bound):
        return tuple(np.linspace(0,b,n) if b else np.array([0.]) for b in bound)+(np.linspace(-1,1,n),)
    grids.append(grid(B))
    for t in range(npre):
        influence=np.maximum(wh,0)@B
        B=(influence+abs(wi)) if t==j else np.sqrt(influence**2+wi**2)
        B*=1+1e-12
        boxes.append(B.copy());grids.append(grid(B))
    return grids,boxes


def transport(model,j,grids,terminal,rules,stats):
    wi,wh,wo=model;T=len(wo);values=terminal
    for t in reversed(range(len(grids)-1)):
        grid=grids[t];nextgrid=grids[t+1]
        states=np.column_stack([a.ravel() for a in np.meshgrid(*grid,indexing='ij')])
        shape=tuple(len(g) for g in grid)+(2,)
        interp=RegularGridInterpolator(nextgrid,values,bounds_error=True)
        if t==j:
            coords=np.column_stack((np.maximum(states[:,:2]@wh.T+states[:,2,None]*wi,0),states[:,2]))
            values=interp(coords).reshape(shape);stats['interpolation_queries']+=len(coords)
            continue
        k=1+t-int(j<t);repeats=len(grid[0])*len(grid[1]);updated=np.zeros((len(states),2))
        for exceeds,sine,cosine,weights in rules[k]:
            sin=np.tile(sine,(repeats,1));cos=np.tile(cosine,(repeats,1));weight=np.tile(weights,(repeats,1,1))
            h=np.maximum((states[:,:2]@wh.T)[:,None,:]*cos[:,:,None]+wi*sin[:,:,None],0)
            coords=np.concatenate((h,(states[:,2,None]*cos)[:,:,None]),axis=2).reshape(-1,3)
            # All nodes, including physically unreachable collocation states, must
            # map into the next box. No clipping or extrapolation is permitted.
            stats['max_box_excess']=max(stats['max_box_excess'],float(np.max(coords[:,:2]-np.array([nextgrid[0][-1],nextgrid[1][-1]]))))
            evaluated=interp(coords).reshape(len(states),sine.shape[1],2)
            if exceeds:updated[:,0]+=np.sum(evaluated[:,:,1]*weight[:,:,0],axis=1)
            else:updated+=np.sum(evaluated*weight,axis=1)
            stats['interpolation_queries']+=len(coords);stats['angle_nodes']+=len(coords)
        values=updated.reshape(shape)
    initial=RegularGridInterpolator(grids[0],values,bounds_error=True)(np.array([[0.,0.,-1.],[0.,0.,1.]]))[:,0]
    stats['interpolation_queries']+=2
    qminus=2*(T+1)/(T*2**T);qplus=2/T-qminus
    return float((initial[0]*qminus+initial[1]*qplus)/2),initial.tolist()


def main():
    started=time.process_time();deadline=started+110
    data=np.load(HERE/'LOCALPILOT_H2_T7_seed0.npz');model=tuple(data[k].astype(float) for k in ('Wi','Wh','Wo'))
    T=len(model[2]);labels=nonvanishing_certificate(model)['nonzero_state_output_labels']
    result={'classification':'deterministic rank-conditioned collocation; no error certificate',
            'model':'LOCALPILOT_H2_T7_seed0','possible_labels':labels,'rows':[],
            'sources':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                [TRANSPORT/'doob.py',TRANSPORT/'backward.py',FUND/'target_tail.py']},
            'model_input_samples':0}
    for n in (3,5):
        kernelwork=defaultdict(float);clock=time.process_time();cache={};grids={};boxes={}
        try:
            for j in labels:
                grids[j],boxes[j]=stage_grids(model,j,n)
                cache[j]=target_kernel_grid(model,j,grids[j][-1],kernelwork,deadline)
            kernel_cpu=time.process_time()-clock
            np.savez(HERE/f'T7_terminal_grid{n}.npz',**{f'j{j}':v for j,v in cache.items()})
            control=(model[0],model[1],np.zeros_like(model[2]));controlgrids,_=stage_grids(control,0,n)
            controlwork=defaultdict(float)
            actual_constant=target_kernel_grid(control,0,controlgrids[-1],controlwork,deadline,constant=True)
            for nq in (32,64):
                clock=time.process_time();stats=defaultdict(float);rank=RankRules(nq,stats)
                b=grids[labels[0]][-1][2];qterminal=rank.mass(b,T-2,2)
                rules=conditional_rules(b,T,8,rank,stats)
                estimates={};initials={};wmin=1.;wmax=0.
                for j in labels:
                    W=cache[j]/qterminal[None,None,:,:]
                    wmin=min(wmin,float(W.min()));wmax=max(wmax,float(W.max()))
                    estimates[j],initials[j]=transport(model,j,grids[j],W,rules,stats)
                ones,_=transport(control,0,controlgrids,np.ones_like(actual_constant),rules,stats)
                sanity,_=transport(control,0,controlgrids,actual_constant/qterminal[None,None,:,:],rules,stats)
                assert abs(ones-1/T)<1e-12
                assert abs(sanity-1/T)<1e-6
                result['rows'].append({'grid':n,'rank_nodes':nq,'angles_per_rank_interval':8,
                    'accuracy':sum(estimates.values()),'contributions':estimates,'initial_sign_values':initials,
                    'boxes':{str(j):[v.tolist() for v in boxes[j]] for j in labels},
                    'terminal_W_min':wmin,'terminal_W_max':wmax,
                    'constant_W_error':ones-1/T,'constant_readout_error':sanity-1/T,
                    'kernel_cpu_seconds':kernel_cpu,'kernel_work':dict(kernelwork),'kernel_reused':nq==64,
                    'extra_control_work':dict(controlwork),'transport_work':dict(stats),
                    'transport_cpu_seconds':time.process_time()-clock})
                (HERE/'finite_transport_results.json').write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps({'grid':n,'rank_nodes':nq,'accuracy':sum(estimates.values()),'constant_error':sanity-1/T}),flush=True)
        except (TimeoutError,ArithmeticError,ValueError,AssertionError) as error:
            result['rows'].append({'grid':n,'failure_type':type(error).__name__,'failure':str(error),'partial_kernel_work':dict(kernelwork)})
            break
    result['cpu_seconds_excluding_imports']=time.process_time()-started
    (HERE/'finite_transport_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cpu_seconds':result['cpu_seconds_excluding_imports'],'rows':len(result['rows'])}))
if __name__=='__main__':main()
