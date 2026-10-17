"""T48 floating geometry + T51 bounded positive R4 mass; independent of tail route."""
import argparse,hashlib,json,time,types
from pathlib import Path
import numpy as np
import torch
from stable_volume import mass


def main():
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint');parser.add_argument('--geometry',required=True);a=parser.parse_args()
    source=Path(a.geometry).read_bytes();geo=types.ModuleType('pinned_geometry');exec(compile(source,a.geometry,'exec'),geo.__dict__)
    output=Path(__file__).with_name('cone_results.json');start=time.process_time()
    data=dict(geometry_path=a.geometry,geometry_sha256=hashlib.sha256(source).hexdigest(),checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),models=[],classification='deterministic bounded volume integration conditional on floating geometry and arithmetic')
    w=torch.load(a.checkpoint,weights_only=True,map_location='cpu')
    base=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));base=(base[0].ravel(),*base[1:])
    rng=np.random.default_rng(51001);models=[base]+[tuple(w+sig*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in base) for sig in (.0025,.02)]
    def save():
        data['total_cpu']=time.process_time()-start;output.write_text(json.dumps(data,indent=2)+'\n')
    for sig,model in zip((0.,.0025,.02),models):
        begin=time.process_time();g=geo.Geometry(seconds=min(25,110-(time.process_time()-start)))
        cells=g.gates(model);correct=g.correct(model,cells);rays=g.simplices(correct);gate_rays=g.simplices([v for v,B in cells])
        wi,wh,wo=model;affine_error=0.
        for vertices,B in cells:
            x=vertices.mean(axis=0);h=np.zeros(len(wi))
            for value in x:h=np.maximum(wh@h+wi*value,0)
            affine_error=max(affine_error,float(np.max(np.abs(h-B@x))))
        row=dict(sigma=sig,geometry_cpu=time.process_time()-begin,counts=g.counts.copy(),correct_simplices=len(rays),gate_simplices=len(gate_rays),correct_determinant_sum=float(abs(np.linalg.det(rays)).sum()),gate_determinant_sum=float(abs(np.linalg.det(gate_rays)).sum()),max_affine_error=affine_error,ray_l1_error=float(np.max(abs(np.sum(abs(rays),axis=1)-1))),same_orthant_rays=bool(np.all(np.min(rays,axis=2)*np.max(rays,axis=2)>=-1e-14)),rules=[])
        data['models'].append(row);save()
        for n in (12,16,20):
            begin=time.process_time();values=[]
            for i,r in enumerate(rays):
                values.append(mass(r,n))
                if i%128==0 and time.process_time()-start>115:raise TimeoutError('cycleCPUcap')
            values=np.array(values)
            row['rules'].append(dict(order=n,estimate=float(values[:,0].sum()),integration_only_bound=float(values[:,1].sum()),cpu_seconds=time.process_time()-begin,quadrature_nodes=len(rays)*n**3))
            save();print(sig,n,row['rules'][-1],flush=True)
        begin=time.process_time();values=np.array([mass(r,16) for r in gate_rays]);row['gate_mass']=dict(estimate=float(values[:,0].sum()),integration_only_bound=float(values[:,1].sum()),cpu_seconds=time.process_time()-begin,quadrature_nodes=len(gate_rays)*16**3);save()
        print('gate',sig,row['gate_mass'],'affine',affine_error,flush=True)
    save()

if __name__=='__main__':main()
