"""T65: stage/target dependent curved boxes; numerical collocation, not certified."""
from collections import defaultdict
import hashlib,json,time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from finite_transport import HERE,TRANSPORT,FUND,RankRules,conditional_rules
from envelope import nonvanishing_certificate
from backward import target_probability


def width(b,box):
    D,C=box
    b=np.asarray(b)
    return abs(b)[...,None]*D+np.sqrt(np.maximum(0,1-b*b))[...,None]*C


def hidden_from(states,box):
    return states[...,:2]*width(states[...,2],box)


def coordinates_from(h,b,box,stats):
    B=width(b,box)
    zero=B==0
    if np.any(np.where(zero,h,0)!=0):raise ArithmeticError('nonzero hidden state at zero box width')
    q=np.divide(h,B,out=np.zeros_like(h),where=~zero)
    stats['zero_width_coordinates']+=int(np.sum(zero))
    stats['max_box_excess']=max(stats['max_box_excess'],float(np.max(q-1)))
    if np.any(q<0) or np.any(q>1):raise ArithmeticError('query outside curved box')
    return np.concatenate((q,np.asarray(b)[...,None]),axis=-1)


def stage_grids(model,j,n,nb=None):
    nb=n if nb is None else nb
    wi,wh,wo=model;T=len(wo);npre=T-2 if j<T-2 else T-3
    A=np.maximum(wh,0);D=np.zeros(2);C=np.zeros(2);boxes=[(D.copy(),C.copy())]
    grids=[(np.array([0.]),np.array([0.]),np.linspace(-1,1,nb))]
    for t in range(npre):
        if t==j:D,C=A@D+abs(wi),A@C
        else:D,C=A@D,np.sqrt((A@C)**2+wi**2)
        # Explicit outward rounding margin, retained in saved coefficients.
        D*=1+1e-12;C*=1+1e-12
        boxes.append((D.copy(),C.copy()))
        grids.append((np.linspace(0,1,n),np.linspace(0,1,n),np.linspace(-1,1,nb)))
    return grids,boxes


def terminal(model,j,grid,box,stats,deadline,cache):
    out=np.zeros(tuple(len(g) for g in grid)+(2,))
    for i,x in enumerate(grid[0]):
        for k,y in enumerate(grid[1]):
            for l,b in enumerate(grid[2]):
                h=hidden_from(np.array([x,y,b]),box)
                for count in (0,1):
                    key=(j,float(h[0]),float(h[1]),float(b),count)
                    if key in cache:p=cache[key];stats['terminal_cache_hits']+=1
                    else:
                        if time.process_time()>deadline:raise TimeoutError('terminal deadline')
                        p,work=target_probability(model,h,float(b),count,j);cache[key]=p
                        stats['new_terminal_calls']+=1
                        for name,value in work.items():
                            if isinstance(value,(float,int)) and name not in ('partition_mass','target_index','prefix_dimension'):stats['terminal_'+name]+=value
                    out[i,k,l,count]=p
    # At collapsed b endpoints every q node must represent exactly the same state.
    for l,b in enumerate(grid[2]):
        B=width(b,box)
        for axis in (0,1):
            if B[axis]==0:
                section=out[:,:,l,:]
                if np.max(abs(section-np.expand_dims(np.take(section,0,axis=axis),axis)))!=0:
                    raise AssertionError('degenerate terminal values disagree')
                stats['collapsed_terminal_axes_checked']+=1
    return out


def transport(model,j,grids,boxes,terminal_values,rules,stats):
    wi,wh,wo=model;T=len(wo);values=terminal_values
    for t in reversed(range(len(grids)-1)):
        grid=grids[t];states=np.column_stack([v.ravel() for v in np.meshgrid(*grid,indexing='ij')])
        hidden=hidden_from(states,boxes[t]);shape=tuple(len(g) for g in grid)+(2,)
        interp=RegularGridInterpolator(grids[t+1],values,bounds_error=True)
        if t==j:
            h=np.maximum(hidden@wh.T+states[:,2,None]*wi,0)
            coords=coordinates_from(h,states[:,2],boxes[t+1],stats)
            values=interp(coords).reshape(shape);stats['interpolation_queries']+=len(coords)
            continue
        k=1+t-int(j<t);repeats=len(grid[0])*len(grid[1]);updated=np.zeros((len(states),2))
        for exceeds,sine,cosine,weights in rules[k]:
            sin=np.tile(sine,(repeats,1));cos=np.tile(cosine,(repeats,1));weight=np.tile(weights,(repeats,1,1))
            h=np.maximum((hidden@wh.T)[:,None,:]*cos[:,:,None]+wi*sin[:,:,None],0)
            coords=coordinates_from(h,states[:,2,None]*cos,boxes[t+1],stats).reshape(-1,3)
            evaluated=interp(coords).reshape(len(states),sine.shape[1],2)
            stats['min_interpolated']=min(stats['min_interpolated'],float(evaluated.min()))
            stats['max_interpolated']=max(stats['max_interpolated'],float(evaluated.max()))
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
    T=len(model[2]);cert=nonvanishing_certificate(model);labels=cert['nonzero_state_output_labels']
    stats=defaultdict(float)
    check=np.array([[0,0,-1],[1,0,0],[0,1,0],[1,1,.7],[.4,.8,-.4]],float)
    box=(np.array([.2,.3]),np.array([2.,3.]))
    error=float(np.max(abs(coordinates_from(hidden_from(check,box),check[:,2],box,stats)-check)))
    assert error<1e-14
    collapsed=(np.zeros(2),np.ones(2))
    assert np.array_equal(coordinates_from(np.zeros((2,2)),np.array([-1.,1.]),collapsed,stats)[:,:2],np.zeros((2,2)))
    result={'classification':'deterministic curved-box collocation; no error certificate','T':T,'labels':labels,
        'coordinate_inverse_error':error,'coordinate_check_work':dict(stats),'rows':[],'model_input_samples':0,
        'checkpoint_sha256':hashlib.sha256((HERE/'LOCALPILOT_H2_T7_seed0.npz').read_bytes()).hexdigest(),
        'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TRANSPORT/'doob.py',TRANSPORT/'backward.py',FUND/'target_tail.py']}}
    cache={}
    for n in (3,5):
        clock=time.process_time();stats=defaultdict(float);rank=RankRules(64,stats)
        b=np.linspace(-1,1,n);rules=conditional_rules(b,T,8,rank,stats)
        constant=(model[0],model[1],np.zeros_like(model[2]));q=np.zeros((n,2));qwork=defaultdict(float)
        for ib,bv in enumerate(b):
            for count in (0,1):
                q[ib,count],work=target_probability(constant,np.zeros(2),float(bv),count,0);qwork['calls']+=1
                for key,value in work.items():
                    if isinstance(value,(int,float)) and key not in ('partition_mass','target_index','prefix_dimension'):qwork[key]+=value
        estimates={};initials={};coefficients={};wmin=1.;wmax=0.
        try:
            for j in labels:
                grids,boxes=stage_grids(model,j,n)
                raw=terminal(model,j,grids[-1],boxes[-1],stats,deadline,cache)
                W=raw/q[None,None,:,:];wmin=min(wmin,float(W.min()));wmax=max(wmax,float(W.max()))
                if wmin < -1e-10 or wmax>1+1e-8:raise ArithmeticError(f'conditional range [{wmin},{wmax}]')
                estimates[j],initials[j]=transport(model,j,grids,boxes,W,rules,stats)
                coefficients[j]=[{'D':D.tolist(),'C':C.tolist()} for D,C in boxes]
            grids,boxes=stage_grids(model,0,n)
            ones=np.ones(tuple(len(g) for g in grids[-1])+(2,))
            control,_=transport(constant,0,grids,boxes,ones,rules,stats)
            assert abs(control-1/T)<1e-12
            result['rows'].append({'grid':n,'rank_nodes':64,'angles':8,'accuracy':sum(estimates.values()),
                'contributions':estimates,'initial_sign_values':initials,'box_coefficients':coefficients,
                'terminal_W_min':wmin,'terminal_W_max':wmax,'constant_W_error':control-1/T,
                'work':dict(stats),'rank_tail_work':dict(qwork),'cpu_seconds':time.process_time()-clock})
            print(json.dumps({'grid':n,'accuracy':sum(estimates.values()),'new_terminal_calls':stats['new_terminal_calls'],'cpu':time.process_time()-clock}),flush=True)
        except (ArithmeticError,ValueError,TimeoutError,AssertionError) as error:
            result['rows'].append({'grid':n,'failure_type':type(error).__name__,'failure':str(error),'partial_work':dict(stats)})
            result['truncated']=True;break
        (HERE/'curved_results.json').write_text(json.dumps(result,indent=2)+'\n')
    result['terminal_unique_states']=len(cache);result['cpu_seconds_excluding_imports']=time.process_time()-started
    (HERE/'curved_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cpu':result['cpu_seconds_excluding_imports'],'terminal_unique_states':len(cache)}))

if __name__=='__main__':main()
