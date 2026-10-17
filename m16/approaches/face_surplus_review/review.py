"""T82: new stencil statistics and geometry-only knot rescue; no accuracy truth."""
import hashlib,importlib.util,json,math,sys,time,resource
from pathlib import Path
import numpy as np
from scipy.special import ndtr,ndtri

HERE=Path(__file__).parent;ROOT=Path('/workspace/clean-neurons');sys.path.insert(0,str(ROOT/'m16'))
from approaches.sci_tao_v2 import core
spec=importlib.util.spec_from_file_location('t82_compiled_reference',ROOT/'m16/approaches/sci_tao_v2/compiled.py')
compiled=importlib.util.module_from_spec(spec);spec.loader.exec_module(compiled)
compiled.__file__=str(HERE/'data/compiled_reference.py') # own precompiled artifact; no shared edits
CASES=((5,.0025,'23',2),(5,.02,'12',2),(10,.0025,'23',2),(10,.02,'12',7))
SHAS={5:'c026d6be1a44e4004d009443b768411e046e75ef42090e525b5ac8a3719e7040',10:'498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb'}


def exponent(values,first=4):
    x=np.arange(first,9,dtype=float);y=np.log2(np.maximum(np.asarray(values[first-1:]),1e-300))
    return float(-np.dot(x-x.mean(),y-y.mean())/np.dot(x-x.mean(),x-x.mean()))


def stencil_indices(grid,level,selection):
    stride=2**(8-level);left=np.arange(0,256,2*stride);middle=left+stride;right=left+2*stride
    keep=np.ones(len(left),bool)
    if selection=='midpoint':keep=(grid[middle]>=.05)&(grid[middle]<=.95)
    if selection=='whole':keep=(grid[left]>=.05)&(grid[right]<=.95)
    return left[keep],middle[keep],right[keep]


def line_statistics(values,grid,selection='all'):
    levels=[]
    for level in range(1,9):
        l,m,r=stencil_indices(grid,level,selection)
        delta=values[:,m]-.5*values[:,l]-.5*values[:,r]
        if not len(l):levels.append(dict(level=level,count=0));continue
        energy=np.sum(delta*delta,axis=1)/len(l);absolute=np.sum(abs(delta),axis=1)/len(l)
        levels.append(dict(level=level,count=len(l),energy=energy.tolist(),absolute=absolute.tolist()))
    return aggregate(levels)


def aggregate(levels):
    for row in levels:
        if not row['count']:continue
        energy=np.asarray(row['energy']);absolute=np.asarray(row['absolute']);n=len(energy);k=max(1,math.ceil(.05*n))
        row.update(rms=float(np.sqrt(energy.mean())),l1=float(absolute.mean()),median_line_rms=float(np.median(np.sqrt(energy))),
            top5_energy_share=float(np.sort(energy)[-k:].sum()/max(energy.sum(),1e-300)))
    out=dict(levels=levels)
    for metric in ('rms','l1'):
        vals=[v.get(metric,float('nan')) for v in levels]
        out[metric+'_exponent_4_8']=exponent(vals,4);out[metric+'_exponent_5_8']=exponent(vals,5)
    return out


def split_statistics(panels):
    levels=[];n=len(panels)
    for level in range(1,9):
        energy=np.zeros(n);absolute=np.zeros(n);counts=[]
        for i,line in enumerate(panels):
            count=0
            for lo,hi,values in line:
                grid=np.linspace(lo,hi,257);l,m,r=stencil_indices(grid,level,'all')
                delta=values[m]-.5*values[l]-.5*values[r]
                energy[i]+=(hi-lo)*float(delta@delta)/len(delta)/.9
                absolute[i]+=(hi-lo)*float(abs(delta).sum())/len(delta)/.9;count+=len(delta)
            counts.append(count)
        levels.append(dict(level=level,count=sum(counts),energy=energy.tolist(),absolute=absolute.tolist(),line_counts=counts))
    return aggregate(levels)


def geometry_knots(anchor,t,family,axis):
    tab=core.tables(t)[0][family];s=float(np.interp(anchor[1],tab[1],core.geometry._g));pu=float(ndtr(s))
    if family=='23':
        assert axis==2
        o2=float(ndtri(max(anchor[3:])*pu))
        candidates=[(float(ndtr(4*s-3*o2))-pu)/(1-pu),(float(ndtr(s+25.5/math.sqrt(2)))-pu)/(1-pu)]
    else:
        assert axis>=2
        other=[anchor[i] for i in range(2,t) if i!=axis]
        candidates=[max(other),float(ndtr(s-25.5/math.sqrt(2)))/pu]
    return [.05]+sorted(set(v for v in candidates if .05<v<.95))+[.95]


def controls():
    grid=np.linspace(1e-5,1-1e-5,257);q=line_statistics((grid*grid)[None,:],grid,'whole')
    assert abs(q['rms_exponent_4_8']-2)<1e-11
    for level in range(1,9):
        l,m,r=stencil_indices(grid,level,'whole');assert np.all(grid[l]>=.05) and np.all(grid[r]<=.95)
    hinge=.371;full=np.linspace(.05,.95,257);f=np.maximum(full-hinge,0)
    panel=[(lo,hi,np.maximum(np.linspace(lo,hi,257)-hinge,0)) for lo,hi in ((.05,hinge),(hinge,.95))]
    split=split_statistics([panel]);old=line_statistics(f[None,:],full)
    equal=[(lo,hi,np.maximum(np.linspace(lo,hi,257)-hinge,0)) for lo,hi in ((.05,.5),(.5,.95))]
    equal_stats=split_statistics([equal]);assert max(r['rms'] for r in split['levels'])<1e-15
    return dict(quadratic=q,hinge_unsplit=old,hinge_split=split,strict_count_level4=len(stencil_indices(grid,4,'whole')[0]),
        midpoint_count_level4=len(stencil_indices(grid,4,'midpoint')[0]),hinge_equal_count_panels=equal_stats)


def model(t,sigma):
    import torch
    path=HERE/'data'/('H16_T5_seed0.pth' if t==5 else 'H16_T10_original.pth')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==SHAS[t]
    state=torch.load(path,map_location='cpu',weights_only=True)
    arrays=[state[k].detach().numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight')]
    arrays[0]=arrays[0][:,0];rng=np.random.default_rng(1)
    return tuple(a+sigma*np.sqrt(np.mean(a*a))*rng.standard_normal(a.shape) for a in arrays)


def knot_controls(anchors,t,family,axis):
    residuals=[];active=[]
    for anchor in anchors:
        cuts=geometry_knots(anchor,t,family,axis)
        for v in cuts[1:-1]:
            u=anchor.copy();u[axis]=v;y,a,b,tau=core.face_points(t,family,u[None,:]);s=float(y[0,a[0]])
            ordered=np.sort(y[0]);
            if family=='23':
                upper=math.sqrt(2)*(ordered[-1]-s)/3;lower=math.sqrt(2)*(s-ordered[-4])
                # Candidate is either branch equality or first-branch cap.
                residuals.append(min(abs(upper-lower),abs(upper-8.5)))
                active.append(bool(abs(upper-lower)<1e-9 and upper<=8.5))
            else:
                pu=float(ndtr(s));others=[anchor[i] for i in range(2,t) if i!=axis]
                value=float(ndtri(v*pu));other=float(ndtri(max(others)*pu))
                residuals.append(min(abs(value-other),abs(math.sqrt(2)*(s-value)/3-8.5)))
                active.append(bool(abs(value-other)<1e-9 and math.sqrt(2)*(s-value)/3<=8.5))
    maximum=max(residuals,default=0.);assert maximum<1e-9
    return dict(candidate_count=len(residuals),max_equation_residual=maximum,active_max_min_switches=sum(active))


class FaceOracle:
    def __init__(self,weights,t,family):self.weights=weights;self.t=t;self.family=family;self.faces=0;self.fpe=0.;self.geometry_cpu=0.;self.walker_cpu=0.;self.max_cells=0
    def __call__(self,u):
        output=[]
        for start in range(0,len(u),4096):
            tick=time.process_time();y,a,b,tau=core.face_points(self.t,self.family,u[start:start+4096]);self.geometry_cpu+=time.process_time()-tick
            tick=time.process_time();f,c,k=compiled.walk_mass(self.weights,y,a,b,tau,self.family);self.walker_cpu+=time.process_time()-tick
            output.append(f);self.faces+=len(f);self.fpe+=float(c.sum());self.max_cells=max(self.max_cells,int(k.max()))
        return np.concatenate(output)
    def report(self):return {k:v for k,v in vars(self).items() if k not in ('weights','t','family')}


def run():
    start=time.process_time();out=dict(protocol='810887c',controls=controls(),cases=[],model_sha256=SHAS)
    out['source_sha256']={name:hashlib.sha256((ROOT/'m16/approaches/sci_tao_v2'/name).read_bytes()).hexdigest() for name in ('core.py','compiled.py','walker.cpp')}
    for t,sigma,family,axis in CASES:
        tick=time.process_time();weights=model(t,sigma);anchors=np.random.default_rng(820001+100*t+int(family)+axis).random((32,t))
        oracle=FaceOracle(weights,t,family);row=dict(t=t,sigma=sigma,family=family,axis=axis)
        row['knot_controls']=knot_controls(anchors,t,family,axis)
        arrays={};raw={}
        for name,lo,hi in (('full',1e-5,1-1e-5),('interior',.05,.95)):
            grid=np.linspace(lo,hi,257);u=np.repeat(anchors,257,axis=0);u[:,axis]=np.tile(grid,len(anchors));f=oracle(u).reshape(32,257)
            arrays[name]=f;raw[name+'_grid']=grid;raw[name+'_values']=f
            selections=('all','midpoint','whole') if name=='full' else ('all',)
            for selection in selections:row[name+'_'+selection]=line_statistics(f,grid,selection)
        # Different backend, same validated walker formulation: explicit scope.
        grid=raw['full_grid'][::16];u=np.repeat(anchors[:1],len(grid),axis=0);u[:,axis]=grid
        y,a,b,tau=core.face_points(t,family,u);f,c,k=core.walk_mass(weights,y,a,b,tau,family)
        row['python_backend_max_difference']=float(np.max(abs(f-arrays['full'][0,::16])));row['python_validation_fpe']=float(c.sum())
        panels=[];all_u=[];specs=[]
        for i,anchor in enumerate(anchors):
            cuts=geometry_knots(anchor,t,family,axis);line=[]
            for lo,hi in zip(cuts[:-1],cuts[1:]):
                u=np.repeat(anchor[None,:],257,axis=0);u[:,axis]=np.linspace(lo,hi,257);all_u.append(u);specs.append((i,lo,hi));line.append((lo,hi))
            panels.append([])
        mass=oracle(np.concatenate(all_u)).reshape(len(specs),257)
        for (i,lo,hi),f in zip(specs,mass):panels[i].append((lo,hi,f))
        row['split_interior']=split_statistics(panels);row['panel_counts']=[len(p) for p in panels]
        # Equal work control: same number of panels on each anchor, equal widths.
        uniform=[[] for _ in anchors];uniform_u=[];uniform_specs=[]
        for i,anchor in enumerate(anchors):
            cuts=np.linspace(.05,.95,len(panels[i])+1)
            for lo,hi in zip(cuts[:-1],cuts[1:]):
                u=np.repeat(anchor[None,:],257,axis=0);u[:,axis]=np.linspace(lo,hi,257)
                uniform_u.append(u);uniform_specs.append((i,lo,hi))
        uniform_mass=oracle(np.concatenate(uniform_u)).reshape(len(uniform_specs),257)
        for (i,lo,hi),f in zip(uniform_specs,uniform_mass):uniform[i].append((lo,hi,f))
        row['equal_count_interior']=split_statistics(uniform)
        row['oracle']=oracle.report();row['cpu_seconds']=time.process_time()-tick
        raw.update(anchors=anchors,panel_specs=np.array(specs),panel_values=mass,uniform_specs=np.array(uniform_specs),uniform_values=uniform_mass)
        tag=f'T{t}_sigma{sigma}_R{family}_axis{axis}'
        np.savez_compressed(HERE/'data'/f'{tag}.npz',**raw)
        out['cases'].append(row);out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n')
        print(tag,'whole',row['full_whole']['rms_exponent_4_8'],'interior',row['interior_all']['rms_exponent_4_8'],
            'split',row['split_interior']['rms_exponent_4_8'],'equal',row['equal_count_interior']['rms_exponent_4_8'],
            'backend',row['python_backend_max_difference'],'faces',oracle.faces,'cpu',row['cpu_seconds'],flush=True)
        if out['cpu_seconds']>100:raise RuntimeError('cycle cap; partial results retained')


if __name__=='__main__':run()
