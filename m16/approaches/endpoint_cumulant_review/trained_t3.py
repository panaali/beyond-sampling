"""Gate/rank cone moments and an endpoint-only trained H2T3 comparison."""
import time
START=time.process_time()
import dataclasses,hashlib,importlib.util,itertools,json,math,resource
from pathlib import Path
import numpy as np
from scipy.special import roots_legendre,gamma
import torch
import controls as U

HERE=Path(__file__).resolve().parent
CP=Path('/workspace/clean-neurons/work/checkpoints/H2_T3_seed0.pth')
GEOM=Path('/workspace/clean-neurons/m16/approaches/exact_ladder/cones.py')
PINS={str(CP):'47aa8e4b38f6433242b13d89666f01add73923b34123a6d56345e2c210cec1bd',
      str(GEOM):'fc006840adc9513ac711b6c208afe96f4d10c429068d087dac8c674e8e6ee948'}
for p,digest in PINS.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest
spec=importlib.util.spec_from_file_location('endpoint_gate_geometry',GEOM);C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)


def measure(Q,ops):
    R,Z=Q;value=0.
    for ids in C.triangulate(R,Z,ops):
        V=R[list(ids)];mass=float(C._p3(C._corr(V[None],ops))[0]);ops.add(3*C.TRANS+8,'solid_angle')
        assert mass>=-1e-12
        value+=mass
    return value


def geometry(M):
    ops=C.Ops();counts=[0,0,0];triangles=[];exact_labels=np.zeros(3);atom=np.zeros(3);truth=0.;nearzero=0;ties=0;annihilated=0
    for R,Z,A,mask in C.iter_cells(M,ops,counts):
        L=M[2]@A;ops.add(2*3*len(M[0])*3,'readout_map')
        for perm in itertools.permutations(range(3)):
            Q=(R,Z)
            for k in range(2):
                normal=np.eye(3)[perm[k]]-np.eye(3)[perm[k+1]]
                Q,_=C.clip(*Q,normal,60+k,ops,want=(True,False))
                if Q is None:break
            if Q is None:continue
            target=perm[1];mass=measure(Q,ops);exact_labels[target]+=mass
            if not A.any():atom[target]+=mass
            if A.any() and not L.any():annihilated+=1
            correct=Q
            for k in range(3):
                if k==target:continue
                normal=L[target]-L[k]
                if not normal.any():
                    ties+=1
                    if k<target:correct=None;break
                else:
                    if np.max(abs(normal))<=1e-13*np.max(abs(L)):nearzero+=1
                    correct,_=C.clip(*correct,normal,70+k,ops,want=(True,False))
                    if correct is None:break
            if correct is not None:truth+=measure(correct,ops)
            for signs in itertools.product([-1,1],repeat=3):
                Qs=Q
                for k,s in enumerate(signs):
                    Qs,_=C.clip(*Qs,s*np.eye(3)[k],90+k,ops,want=(True,False))
                    if Qs is None:break
                if Qs is None:continue
                Rs,Zs=Qs
                for ids in C.triangulate(Rs,Zs,ops):
                    V=Rs[list(ids)];assert np.min(V@V.T)>=-1e-9
                    triangles.append((target,V,L,not A.any()))
    return triangles,dict(gate_counts=counts,triangles=len(triangles),rank_masses=exact_labels.tolist(),
        hzero_atom_joint=atom.tolist(),accuracy=truth,geometry_ops=ops.n,geometry_ops_parts=ops.parts,
        exact_zero_margins=ties,nearzero_nonzero_margins=nearzero,nonzero_hidden_zero_readout_pieces=annihilated)


def moments(triangles,n):
    started=time.process_time();z,w=roots_legendre(n);z=(z+1)/2;w=w/2
    u,v=np.meshgrid(z,z,indexing='ij');weights=(w[:,None]*w[None,:]*(1-u)).ravel()
    b=np.stack([u,(1-u)*v,(1-u)*(1-v)],axis=-1).reshape(-1,3)
    result={k:np.zeros((3,)+(3,)*k) for k in range(5)};nodes=0;dets=[]
    for target,V,L,dead in triangles:
        x=b@V;norm=np.linalg.norm(x,axis=1);det=abs(np.linalg.det(V));dets.append(det)
        assert np.min(norm)>0
        weight=weights*det/(4*np.pi*norm**3);logits=(x@L.T)/norm[:,None]
        result[0][target]+=weight.sum();nodes+=len(weight)
        for k in range(1,5):
            rad=2**(k/2)*gamma((3+k)/2)/gamma(1.5)
            labels='ijkl'[:k];expr='n,'+','.join('n'+s for s in labels)+'->'+labels
            result[k][target]+=rad*np.einsum(expr,weight,*([logits]*k),optimize=True)
    return result,dict(order=n,nodes=nodes,triangles=len(triangles),determinant_min=min(dets),
        cpu_seconds=time.process_time()-started,raw_tensor_entries=sum(3**k for k in range(5)),
        note='Moment evaluations and counts; no certified scalar-FLOP or quadrature-error enclosure.')


def cumulants(joint,label,mass):
    raw={k:joint[k][label]/mass for k in range(1,5)};mu=raw[1];K={1:mu}
    for k in range(2,5):
        ids=np.indices((3,)*k);central=np.zeros((3,)*k)
        for mask in itertools.product([False,True],repeat=k):
            selected=[i for i in range(k) if mask[i]]
            term=raw[len(selected)][tuple(ids[i] for i in selected)] if selected else np.ones((3,)*k)
            for i in range(k):
                if not mask[i]:term=term*mu[ids[i]]
            central+=(-1)**(k-len(selected))*term
        if k==4:
            a,b,c,d=ids;cov=K[2]
            central-=cov[a,b]*cov[c,d]+cov[a,c]*cov[b,d]+cov[a,d]*cov[b,c]
        K[k]=central
    return K


def main():
    state=torch.load(CP,weights_only=True,map_location='cpu');M=(state['rnn.weight_ih_l0'].numpy()[:,0].astype(float),
        state['rnn.weight_hh_l0'].numpy().astype(float),state['linear.weight'].numpy().astype(float))
    begin=time.process_time();triangles,geom=geometry(M);geom['cpu_seconds']=time.process_time()-begin
    assert max(abs(np.array(geom['rank_masses'])-1/3))<1e-9
    allrules=[];old=None;accepted=False
    for n in [16,32,64]:
        joint,stats=moments(triangles,n);stats['rank_mass_discrepancy']=float(np.max(abs(joint[0]-geom['rank_masses'])))
        if old is not None:
            differences={k:float(np.max(abs(joint[k]-old[k]))/max(1.,np.max(abs(joint[k])))) for k in joint}
            stats['relative_moment_differences']=differences
            accepted=stats['rank_mass_discrepancy']<=1e-9 and max(differences.values())<=1e-8
        allrules.append(stats);old=joint
        if accepted:break
    out=dict(complete=False,geometry=geom,moment_rules=allrules,moment_reference_accepted=accepted,
        joint_moments={k:v.tolist() for k,v in joint.items()},source_pins=PINS,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),endpoint_cases=[])
    (HERE/'trained_t3.json').write_text(json.dumps(out,indent=2)+'\n')
    assert accepted,'Frozen moment sensitivity criteria failed'
    for split in [False,True]:
        acc={};proj={};cases=[]
        for j in range(3):
            mass=joint[0][j]-(geom['hzero_atom_joint'][j] if split else 0.)
            assert mass>0
            K=cumulants(joint,j,mass);tower={k:U.HTensor(torch.tensor(v,dtype=torch.float64),r=0) for k,v in K.items()}
            result=U.argmax_endpoint_estimate(tower,U.QuadratureCfg(num_nodes=128))
            row=U.jsonable(dataclasses.asdict(result));row.update(target=j,nonatom_conditioning=split,law_mass=mass,
                cumulants={k:v.tolist() for k,v in K.items()},cov_eigenvalues=np.linalg.eigvalsh(K[2]).tolist())
            cases.append(row)
            for method,q in row['q_raw'].items():
                acc[method]=acc.get(method,0)+mass*q[j]+(geom['hzero_atom_joint'][j] if split and j==0 else 0.)
                proj[method]=proj.get(method,0)+mass*row['q_projected'][method][j]+(geom['hzero_atom_joint'][j] if split and j==0 else 0.)
            if time.process_time()-START>95:raise TimeoutError('frozen CPU deadline')
        out['endpoint_cases'].append(dict(atom_separated=split,conditional=cases,accuracy=acc,
            projected_accuracy=proj,bias={k:v-geom['accuracy'] for k,v in acc.items()}))
    out.update(complete=True,cpu_seconds_including_imports=time.process_time()-START,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (HERE/'trained_t3.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['joint_moments','endpoint_cases']},indent=2))
    for row in out['endpoint_cases']:print('atom',row['atom_separated'],'accuracy',row['accuracy'],'bias',row['bias'])


if __name__=='__main__':main()
