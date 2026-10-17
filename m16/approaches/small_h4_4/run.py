import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from estimator import angular_estimate,forward


def main():
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint');parser.add_argument('--orders',default='2,4,8');parser.add_argument('--output',default='results.json');parser.add_argument('--mode',default='octants');parser.add_argument('--rules');a=parser.parse_args()
    start=time.process_time()
    output=Path(__file__).with_name(a.output)
    data=dict(checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),models=[],class_name='deterministic outer quadrature with analytic two-input tail')
    w=torch.load(a.checkpoint,weights_only=True,map_location='cpu')
    base=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));base=(base[0].ravel(),*base[1:])
    noise_rng=np.random.default_rng(51001)
    models=[base]+[tuple(w+sig*np.sqrt(np.mean(w*w))*noise_rng.standard_normal(w.shape) for w in base) for sig in (.0025,.02)]
    for sig,m in zip((0.,.0025,.02),models):
        rng=np.random.default_rng(51002);correct=0;n=262144;clock=time.process_time()
        for i in range(0,n,65536):
            x=rng.standard_normal((min(65536,n-i),4))
            correct+=int((forward(m,x).argmax(1)==np.argsort(x,axis=1)[:,-2]).sum())
        elapsed=time.process_time()-clock;truth=correct/n;se=float(np.sqrt(truth*(1-truth)/n))
        row=dict(sigma=sig,truth_mc=truth,truth_n=n,truth_se=se,sampling_cpu=elapsed,rules=[])
        data['models'].append(row)
        rules=[tuple(map(int,item.split(':'))) for item in a.rules.split(',')] if a.rules else [(int(x),1) for x in a.orders.split(',')]
        for order,subdivision in rules:
            remaining=115-(time.process_time()-start)
            if remaining<1:raise TimeoutError('cycleCPUbudget exhausted')
            p,stats=angular_estimate(m,order,max_cpu=remaining,boundary_mode=a.mode,subdivision=subdivision)
            row['rules'].append(dict(estimate=p,stats=stats,difference_from_mc=p-truth,z=(p-truth)/se,sampling_se_equal_cpu=float(np.sqrt(truth*(1-truth)*elapsed/n/stats['cpu_seconds']))))
            data['total_cpu']=time.process_time()-start
            output.write_text(json.dumps(data,indent=2)+'\n')
            print(sig,order,subdivision,p,stats['cpu_seconds'],truth,se,flush=True)
    print('totalCPU',time.process_time()-start,flush=True)

if __name__=='__main__':main()
