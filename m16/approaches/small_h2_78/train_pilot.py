"""Bounded, reproducible LOCALPILOT checkpoint. Never an AlgZoo-trained model."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np


def train(seq_len=7,seed=0,max_steps=2000,max_cpu=30,heldout_seed=49003,out_dir=None):
    import torch
    torch.set_num_threads(1)
    start=time.process_time();torch.manual_seed(seed)
    rnn=torch.nn.RNN(1,2,nonlinearity='relu',bias=False,batch_first=True)
    readout=torch.nn.Linear(2,seq_len,bias=False)
    parameters=[*rnn.parameters(),*readout.parameters()]
    optimizer=torch.optim.Adam(parameters,lr=.01)
    generator=torch.Generator().manual_seed(seed)
    loss_trace=[];last_loss=None;steps=0;stop='max_steps'
    for step in range(max_steps):
        if time.process_time()-start>=max_cpu-3:
            stop='cpu_budget';break
        x=torch.randn(256,seq_len,1,generator=generator)
        target=torch.argsort(x[:,:,0],dim=1)[:,-2]
        h,_=rnn(x);logits=readout(h[:,-1,:]);loss=torch.nn.functional.cross_entropy(logits,target)
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        steps=step+1;last_loss=float(loss.detach())
        if steps%200==0 or steps==1:loss_trace.append({'step':steps,'loss':last_loss})
    training_cpu=time.process_time()-start
    rng=np.random.default_rng(heldout_seed);N=32768;correct=0
    with torch.no_grad():
        for _ in range(N//4096):
            x=rng.standard_normal((4096,seq_len)).astype(np.float32)
            h,_=rnn(torch.from_numpy(x[:,:,None]));pred=readout(h[:,-1,:]).argmax(1).numpy()
            target=np.argsort(x,axis=1)[:,-2];correct+=int(np.sum(pred==target))
    accuracy=correct/N
    weights={'Wi':rnn.weight_ih_l0.detach().numpy().reshape(-1),
             'Wh':rnn.weight_hh_l0.detach().numpy(),'Wo':readout.weight.detach().numpy()}
    out_dir=Path(out_dir or Path(__file__).parent);out_dir.mkdir(parents=True,exist_ok=True)
    path=out_dir/f'LOCALPILOT_H2_T{seq_len}_seed{seed}.npz';np.savez(path,**weights)
    result={'provenance':'LOCALPILOT, not AlgZoo pretrained or 2^30-sequence training',
            'H':2,'T':seq_len,'seed':seed,'training_input_seed':seed,'heldout_seed':heldout_seed,
            'architecture':'bias-free ReLU torch.nn.RNN plus final linear readout',
            'optimizer':'Adam, lr0.01, PyTorch defaults','loss':'cross entropy, second-largest input position',
            'batch':256,'steps':steps,'training_sequences':256*steps,'max_steps':max_steps,
            'stop_reason':stop,'training_cpu_seconds':training_cpu,'max_cpu_seconds':max_cpu,
            'heldout_sequences':N,'heldout_correct':correct,'heldout_accuracy':accuracy,
            'heldout_standard_error':float(np.sqrt(accuracy*(1-accuracy)/N)),
            'loss_trace':loss_trace,'last_loss':last_loss,'torch_version':torch.__version__,
            'numpy_version':np.__version__,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'total_cpu_seconds':time.process_time()-start}
    path.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return weights,result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seq-len',type=int,default=7)
    p.add_argument('--heldout-seed',type=int,default=49003);p.add_argument('--out-dir')
    args=p.parse_args();train(seq_len=args.seq_len,heldout_seed=args.heldout_seed,out_dir=args.out_dir)
