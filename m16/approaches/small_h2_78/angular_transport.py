"""T61: matched angular-grid comparison for the local T7 pilot."""
from collections import defaultdict
from fractions import Fraction
import json,time
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from envelope import envelope,nonvanishing_certificate
from finite_transport import stage_grids,RankRules,conditional_rules
from backward import target_probability

HERE=Path(__file__).resolve().parent


def hidden_from(sz,box):
    sz=np.asarray(sz);s,z=sz[...,0],sz[...,1]
    return np.stack((s*box[0]*z,s*box[1]*(1-z)),axis=-1)/np.maximum(z,1-z)[...,None]


def coordinates_from(h,box):
    q=np.asarray(h)/np.asarray(box)
    s=np.max(q,axis=-1);den=np.sum(q,axis=-1)
    z=np.divide(q[...,0],den,out=np.full_like(den,.5),where=den!=0)
    return np.stack((s,z),axis=-1)


def make_grids(model,j,n,aligned,preprocessing=None):
    _,boxes=stage_grids(model,j,n);e=envelope(model[2]);cells=e['cells']
    if preprocessing is not None:
        preprocessing['envelope_constructions']+=1
        for key in ('rational_arithmetic_operations','rational_comparisons'):preprocessing[key]+=e[key]
    boundaries=[float(Fraction(cells[i]['left'])) for i in range(1,len(cells))
                if cells[i-1]['winner']!=cells[i]['winner']]
    grids=[];directions=[]
    for B in boxes:
        if not np.any(B):
            grids.append((np.array([0.]),np.array([.5]),np.linspace(-1,1,n)));directions.append([]);continue
        mapped=[B[1]/(B[1]+r*B[0]) for r in boundaries]
        angular=np.array(sorted({*np.linspace(0,1,n),*mapped}))
        if not aligned:angular=np.linspace(0,1,len(angular))
        grids.append((np.linspace(0,1,n),angular,np.linspace(-1,1,n)));directions.append(mapped)
    return grids,boxes,directions


def terminal(model,j,grid,box,stats,deadline,cache):
    values=np.zeros(tuple(len(g) for g in grid)+(2,))
    for ir,r in enumerate(grid[0]):
        for iz,z in enumerate(grid[1] if r else grid[1][:1]):
            h=hidden_from(np.array([r,z]),box)
            for ib,b in enumerate(grid[2]):
                for count in (0,1):
                    key=(j,float(h[0]),float(h[1]),float(b),count)
                    if key in cache:
                        p=cache[key];stats['terminal_cache_hits']+=1
                    else:
                        if time.process_time()>deadline:raise TimeoutError('cycle deadline before terminal probability')
                        p,work=target_probability(model,h,float(b),count,j);cache[key]=p
                        stats['new_terminal_calls']+=1
                        for name,value in work.items():
                            if isinstance(value,(float,int)) and name not in ('partition_mass','target_index','prefix_dimension'):
                                stats['terminal_'+name]+=value
                    values[ir,iz,ib,count]=p
        if not r:values[ir,:,:,:]=values[ir,0,:,:][None,:,:]
    return values


def transport(model,j,grids,boxes,terminal_values,rules,stats):
    wi,wh,wo=model;T=len(wo);values=terminal_values
    for t in reversed(range(len(grids)-1)):
        grid=grids[t];nextgrid=grids[t+1]
        states=np.column_stack([a.ravel() for a in np.meshgrid(*grid,indexing='ij')])
        hidden=hidden_from(states[:,:2],boxes[t]);shape=tuple(len(g) for g in grid)+(2,)
        interp=RegularGridInterpolator(nextgrid,values,bounds_error=True)
        if t==j:
            h=np.maximum(hidden@wh.T+states[:,2,None]*wi,0)
            coords=np.column_stack((coordinates_from(h,boxes[t+1]),states[:,2]))
            values=interp(coords).reshape(shape);stats['interpolation_queries']+=len(coords)
            continue
        k=1+t-int(j<t);repeats=len(grid[0])*len(grid[1]);updated=np.zeros((len(states),2))
        for exceeds,sine,cosine,weights in rules[k]:
            sin=np.tile(sine,(repeats,1));cos=np.tile(cosine,(repeats,1));weight=np.tile(weights,(repeats,1,1))
            h=np.maximum((hidden@wh.T)[:,None,:]*cos[:,:,None]+wi*sin[:,:,None],0)
            sz=coordinates_from(h,boxes[t+1]);coords=np.concatenate((sz,(states[:,2,None]*cos)[:,:,None]),axis=2).reshape(-1,3)
            stats['max_amplitude_excess']=max(stats['max_amplitude_excess'],float(np.max(coords[:,0]-1)))
            evaluated=interp(coords).reshape(len(states),sine.shape[1],2)
            stats['min_interpolated']=min(stats['min_interpolated'],float(evaluated.min()))
            stats['max_interpolated']=max(stats['max_interpolated'],float(evaluated.max()))
            if exceeds:updated[:,0]+=np.sum(evaluated[:,:,1]*weight[:,:,0],axis=1)
            else:updated+=np.sum(evaluated*weight,axis=1)
            stats['interpolation_queries']+=len(coords);stats['angle_nodes']+=len(coords)
        values=updated.reshape(shape)
    initial=RegularGridInterpolator(grids[0],values,bounds_error=True)(np.array([[0.,.5,-1.],[0.,.5,1.]]))[:,0]
    stats['interpolation_queries']+=2
    qminus=2*(T+1)/(T*2**T);qplus=2/T-qminus
    return float((initial[0]*qminus+initial[1]*qplus)/2),initial.tolist()


def main():
    started=time.process_time();deadline=started+110
    data=np.load(HERE/'LOCALPILOT_H2_T7_seed0.npz');model=tuple(data[k].astype(float) for k in ('Wi','Wh','Wo'))
    T=len(model[2]);labels=nonvanishing_certificate(model)['nonzero_state_output_labels']
    # Deterministic inverse checks include both axes, both corners and origin.
    box=np.array([2.,3.]);coords=np.array([[0.,.5],[1.,0.],[1.,1.],[1.,.5],[.4,.2],[.7,.8]])
    inverse_error=float(np.max(abs(coordinates_from(hidden_from(coords,box),box)-coords)))
    assert inverse_error<1e-14
    result={'classification':'deterministic amplitude-angle collocation; no error certificate',
            'T':T,'labels':labels,'inverse_error':inverse_error,'rows':[],'model_input_samples':0}
    cache={};preprocessing=defaultdict(int)
    for n in (3,5):
        b=np.linspace(-1,1,n);rankstats=defaultdict(float);rank=RankRules(64,rankstats)
        rules=conditional_rules(b,T,8,rank,rankstats)
        # Exact rank denominator through the analytic constant-output tail.
        constant=(model[0],model[1],np.zeros_like(model[2]));q=np.zeros((n,2));qwork=defaultdict(float)
        for ib,bv in enumerate(b):
            for count in (0,1):
                q[ib,count],work=target_probability(constant,np.zeros(2),float(bv),count,0)
                qwork['calls']+=1
                for key,value in work.items():
                    if isinstance(value,(int,float)) and key not in ('partition_mass','target_index','prefix_dimension'):qwork[key]+=value
        for aligned in (False,True):
            clock=time.process_time();stats=defaultdict(float);estimates={};initials={};counts={};wmin=1.;wmax=0.
            try:
                for j in labels:
                    grids,boxes,directions=make_grids(model,j,n,aligned,preprocessing)
                    raw=terminal(model,j,grids[-1],boxes[-1],stats,deadline,cache)
                    W=raw/q[None,None,:,:]
                    wmin=min(wmin,float(W.min()));wmax=max(wmax,float(W.max()))
                    if wmin < -1e-10 or wmax>1+1e-8:raise ArithmeticError(f'conditional range [{wmin},{wmax}]')
                    estimates[j],initials[j]=transport(model,j,grids,boxes,W,rules,stats)
                    counts[j]={'radial_nodes':len(grids[-1][0]),'angular_nodes':len(grids[-1][1]),
                        'target_nodes':len(grids[-1][2]),'readout_directions_added':len(directions[-1]) if aligned else 0,
                        'extra_angular_nodes_vs_radial':len(grids[-1][1])-n,
                        'terminal_grid_points':int(np.prod([len(g) for g in grids[-1]])),
                        'terminal_angular_values':grids[-1][1].tolist()}
                controlgrids,controlboxes,_=make_grids(model,0,n,aligned,preprocessing)
                ones=np.ones(tuple(len(g) for g in controlgrids[-1])+(2,))
                control,_=transport(model,0,controlgrids,controlboxes,ones,rules,stats)
                assert abs(control-1/T)<1e-12
                row={'radial_target_size':n,'aligned':aligned,'accuracy':sum(estimates.values()),
                    'contributions':estimates,'initial_sign_values':initials,'node_counts':counts,
                    'terminal_W_min':wmin,'terminal_W_max':wmax,'constant_W_error':control-1/T,
                    'work':dict(stats),'rank_work':dict(rankstats),'rank_work_reused':aligned,
                    'rank_tail_work':dict(qwork),'rank_tail_reused':aligned,'cpu_seconds':time.process_time()-clock}
                result['rows'].append(row)
                print(json.dumps({'n':n,'aligned':aligned,'accuracy':row['accuracy'],'new_calls':stats['new_terminal_calls'],'cpu':row['cpu_seconds']}),flush=True)
            except (ArithmeticError,ValueError,TimeoutError,AssertionError) as error:
                result['rows'].append({'n':n,'aligned':aligned,'failure_type':type(error).__name__,'failure':str(error),'partial_work':dict(stats)})
                result['truncated']=True
                break
            (HERE/'angular_results.json').write_text(json.dumps(result,indent=2)+'\n')
        if result.get('truncated'):break
    result['cpu_seconds_excluding_imports']=time.process_time()-started
    result['terminal_unique_states']=len(cache)
    result['readout_preprocessing']={**dict(preprocessing),'sorting_and_bit_cost_excluded':True}
    (HERE/'angular_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'total_cpu':result['cpu_seconds_excluding_imports'],'unique_terminal_states':len(cache)}))
if __name__=='__main__':main()
