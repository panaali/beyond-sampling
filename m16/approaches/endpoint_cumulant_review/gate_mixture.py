"""Frozen gate/rank mixture with direct cone support tests before endpoint use."""
import time
START=time.process_time()
import dataclasses,hashlib,itertools,json,resource
from pathlib import Path
import numpy as np
import torch
import trained_t3 as T

HERE=Path(__file__).resolve().parent
C=T.C


def winner(R,L,ops):
    for j in range(3):
        valid=True
        for k in range(3):
            if j==k:continue
            d=L[j]-L[k];vals=R@d;ops.add(3+6*len(R),'support')
            if np.any(vals<0) or (k<j and not np.any(vals>0)):
                valid=False;break
        if valid:return j
    return None


def groups(M):
    ops=C.Ops();counts=[0,0,0];out=[]
    for ci,(R,Z,A,mask) in enumerate(C.iter_cells(M,ops,counts)):
        L=M[2]@A;ops.add(36,'readout_map')
        for perm in itertools.permutations(range(3)):
            Q=(R,Z)
            for k in range(2):
                normal=np.eye(3)[perm[k]]-np.eye(3)[perm[k+1]]
                Q,_=C.clip(*Q,normal,60+k,ops,want=(True,False))
                if Q is None:break
            if Q is None:continue
            j=perm[1];fixed=winner(Q[0],L,ops);mass=T.measure(Q,ops)
            assert mass>0
            triangles=[]
            for signs in itertools.product([-1,1],repeat=3):
                Qs=Q
                for k,s in enumerate(signs):
                    Qs,_=C.clip(*Qs,s*np.eye(3)[k],90+k,ops,want=(True,False))
                    if Qs is None:break
                if Qs is None:continue
                Rs,Zs=Qs
                triangles.extend((j,Rs[list(ids)],L,not A.any()) for ids in C.triangulate(Rs,Zs,ops))
            out.append(dict(gate_cell=ci,order=perm,target=j,mass=mass,fixed_winner=fixed,triangles=triangles))
    return out,dict(gate_counts=counts,geometry_ops=ops.n,geometry_ops_parts=ops.parts)


def main():
    state=torch.load(T.CP,weights_only=True,map_location='cpu');M=(state['rnn.weight_ih_l0'].numpy()[:,0].astype(float),
        state['rnn.weight_hh_l0'].numpy().astype(float),state['linear.weight'].numpy().astype(float))
    pieces,geom=groups(M);methods=['E0_product_gaussian','E1_cov1','E2_cov2','E2_k3','E2_full']
    total={m:0. for m in methods};projected=total.copy();fixed_mass=0.;residual_mass=0.;rows=[];moment_nodes=0;endpoint_ops=0
    sums={k:np.zeros((3,)+(3,)*k) for k in range(5)}
    out=dict(complete=False,geometry=geom,components=rows,source_pins=T.PINS,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for group in pieces:
        triangles=group['triangles'];j=group['target'];old,s16=T.moments(triangles,16);joint,s32=T.moments(triangles,32)
        diff={k:float(np.max(abs(joint[k]-old[k]))/max(1.,np.max(abs(joint[k])))) for k in joint}
        massdiff=abs(joint[0][j]-group['mass']);accepted=massdiff<=1e-9 and max(diff.values())<=1e-8
        row={k:v for k,v in group.items() if k!='triangles'};row.update(triangles=len(triangles),moment_rules=[s16,s32],
            mass_discrepancy=massdiff,moment_relative_differences=diff,moment_accepted=accepted)
        moment_nodes+=s16['nodes']+s32['nodes']
        assert accepted,'Component moment criteria failed'
        for k in sums:sums[k]+=joint[k]
        fixed=group['fixed_winner'];mass=group['mass']
        if fixed is not None:
            fixed_mass+=mass
            for m in methods:total[m]+=mass*(fixed==j);projected[m]+=mass*(fixed==j)
        else:
            residual_mass+=mass;K=T.cumulants(joint,j,float(joint[0][j]));tower={k:T.U.HTensor(torch.tensor(v,dtype=torch.float64),r=0) for k,v in K.items()}
            result=T.U.argmax_endpoint_estimate(tower,T.U.QuadratureCfg(num_nodes=128));r=T.U.jsonable(dataclasses.asdict(result))
            row['endpoint']=r;row['cumulants']={k:v.tolist() for k,v in K.items()};endpoint_ops+=r['flops_endpoint_total']
            for m in methods:total[m]+=mass*r['q_raw'][m][j];projected[m]+=mass*r['q_projected'][m][j]
        rows.append(row)
        (HERE/'gate_mixture.json').write_text(json.dumps(out,indent=2)+'\n')
        if time.process_time()-START>95:raise TimeoutError('frozen CPU deadline')
    previous=json.loads((HERE/'trained_t3.json').read_text());truth=previous['geometry']['accuracy']
    reconciliation={k:float(np.max(abs(v-np.array(previous['joint_moments'][str(k)])))/max(1.,np.max(abs(v)))) for k,v in sums.items()}
    assert max(reconciliation.values())<1e-10
    out.update(complete=True,component_count=len(pieces),constant_components=sum(p['fixed_winner'] is not None for p in pieces),
        fixed_winner_mass=fixed_mass,residual_mass=residual_mass,accuracy=total,projected_accuracy=projected,
        truth=truth,bias={k:v-truth for k,v in total.items()},moment_nodes=moment_nodes,endpoint_proxy_ops=endpoint_ops,
        joint_moment_reconciliation=reconciliation,cpu_seconds_including_imports=time.process_time()-START,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (HERE/'gate_mixture.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='components'},indent=2))


if __name__=='__main__':main()
