"""Exact open-chart witnesses and fixed-implementation nominal completion bound."""
from pathlib import Path
import itertools,math,json,time,hashlib
HERE=Path(__file__).parent

def witness(perm,sigma,gapside):
    T=len(perm);v=[0]*T;ordered=([3,2,0] if gapside==1 else [3,1,0])+list(range(-1,-(T-3)-1,-1))
    for i,j in enumerate(perm):v[j]=ordered[i]
    shift=sigma*(T+4)-v[0];return [z+shift for z in v]

def main():
    stamp=time.process_time();checks=[]
    for T in (3,4,5):
        count=0;minimum=10**9
        for perm in itertools.permutations(range(T)):
            for sigma,gapside in itertools.product((-1,1),repeat=2):
                x=witness(perm,sigma,gapside);slacks=[x[perm[q]]-x[perm[q+1]] for q in range(T-1)]+[sigma*x[0],gapside*(2*x[perm[1]]-x[perm[0]]-x[perm[2]])];assert min(slacks)>0
                minimum=min(minimum,min(slacks));count+=1
        assert count==4*math.factorial(T);checks.append(dict(T=T,charts=count,minimum_integer_slack=minimum))
    import torch
    p=Path('/workspace/work/t39_sources/example_checkpoint.pth');digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest=='498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb'
    sd=torch.load(p,map_location='cpu',weights_only=True);wi=sd['rnn.weight_ih_l0'][:,0];H=wi.numel();T=sd['linear.weight'].shape[0];r=int((wi!=0).sum());assert (H,T,r)==(16,10,16)
    fact=math.factorial(T);gate=4*H*fact;first=2*H*r*fact;bound=gate+first;assert bound==2090188800
    first_ray_tests=4*fact*T*H*(T+1)
    out=dict(chart_controls=checks,target=dict(H=H,T=T,nonzero_Wi=r,positive_Wi=int((wi>0).sum()),negative_Wi=int((wi<0).sum()),checkpoint_sha256=digest,charts=4*fact,first_gate_nominal_ops=gate,first_recurrence_nominal_ops=first,completion_prefix_lower_bound=bound,initial_ray_gate_nominal_lower_bound=first_ray_tests,combined_declared_lower_bound=bound+first_ray_tests),scope='Completion of unchanged factorial-chart implementation under its declared charges; no T10 run, deeper work/masses/preprocessing excluded; not a method-family impossibility.',cpu_seconds=time.process_time()-stamp)
    (HERE/'chart_prefix_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
