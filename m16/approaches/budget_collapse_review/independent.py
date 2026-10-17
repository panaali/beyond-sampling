"""Same frozen face integration with distinct scramble namespaces per kind."""
import time
import numpy as np
import engine as E


def faces(M,n,seed):
    T=M[2].shape[0];unit=E.C.fpe_flops(len(M[0]),T);_,Z23,Z12=E.C.tables(T)
    aggregate=0.;walk=0.;counts={};kindcost={};started=time.process_time()
    for kind,Z,offset in [('23',Z23,0),('12',Z12,100000)]:
        U=E.C.sobol(T,n,seed+offset);mass=[];cost=0.
        for lo in range(0,n,4096):
            y,a,b,tau=E.C.face_points(T,kind,U[lo:lo+4096])
            values,cs,cells,work=E.C.walk_mass(M,y,a,b,tau,kind,details=True)
            mass.extend(values.tolist());cost+=float(cs.sum())
            counts['cells']=counts.get('cells',0)+int(cells.sum())
            for name,vals in work.items():counts[name]=counts.get(name,0)+int(vals.sum())
        aggregate+=Z*float(np.mean(mass));walk+=cost;kindcost[kind]=cost
    return dict(estimate=1-aggregate,walk_fpe=walk,node_flops=2*n*100*T,
                variable_fpe=walk+2*n*100*T/unit,kind_walk_fpe=kindcost,counts=counts,
                faces_per_kind=n,cpu_seconds=time.process_time()-started,
                kind_rule_seeds={'23':seed,'12':seed+100000})


def est(M,budget,rng):
    if not np.isfinite(budget) or budget<=0:raise ValueError('positive finite budget required')
    T=M[2].shape[0];pilot=E.faces(M,64,44001);fixed=E.fixed_cost(T,len(M[0]))
    per_pair=pilot['walk_fpe']/64+200*T/fixed['unit']
    n=int(np.floor((budget-fixed['setup_fpe']-pilot['variable_fpe'])/per_pair))
    if n<16:raise ValueError('budget cannot fund frozen setup, pilot and minimum rule')
    result=faces(M,n,int(rng.integers(1,2**30)))
    return result['estimate'],fixed['setup_fpe']+pilot['variable_fpe']+result['variable_fpe']
