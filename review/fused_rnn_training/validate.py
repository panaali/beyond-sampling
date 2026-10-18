"""Fixed CPU interpreter vs independent torch-autograd/update controls.
No GPU or throughput inference. Explicit NumPy libdevice interpreter shim only.
"""
import os
os.environ['TRITON_INTERPRET']='1'
import hashlib,json,math,time,traceback,resource
from pathlib import Path
import numpy as np
import torch
import triton
from triton.runtime import interpreter
P=Path(__file__).resolve().parent

# Triton3.8 interpreter has no extern_elementwise implementation. These exact
# symbol-specific CPU implementations only provide double pow/sqrt schedule ops.
# Actual CUDA libdevice remains unvalidated; no general arbitrary extern fallback.
def install_libdevice_shim():
    original=interpreter.InterpreterBuilder.create_extern_elementwise
    def dispatch(self,libName,libPath,symbol,argList,retType,isPure):
        if symbol=='__nv_pow':return interpreter.TensorHandle(np.power(argList[0].data,argList[1].data),retType)
        if symbol=='__nv_sqrt':return interpreter.TensorHandle(np.sqrt(argList[0].data),retType)
        return original(self,libName,libPath,symbol,argList,retType,isPure)
    interpreter.InterpreterBuilder.create_extern_elementwise=dispatch
    return original


def reference(params,m,v,steps,n_steps,warm,lr0,x,y):
    pp=[p.detach().clone().requires_grad_() for p in params]
    h=torch.zeros((x.shape[0],x.shape[1],16),dtype=torch.float32)
    for t in range(10):h=torch.relu(torch.bmm(h,pp[1].transpose(1,2))+x[:,:,t,None]*pp[0][:,None,:])
    logits=torch.bmm(h,pp[2].transpose(1,2))
    loss=torch.nn.functional.cross_entropy(logits.reshape(-1,10),y.reshape(-1),reduction='none').reshape(x.shape[:2]).mean(1)
    grads=torch.autograd.grad(loss.sum(),pp)
    acc=(logits.argmax(-1)==y).float().mean(1)
    with torch.no_grad():
        lr=(lr0*torch.clamp(steps/warm,max=1)*torch.clamp(.01**(2*steps/n_steps),min=.01)*(steps<n_steps).double()).float()
        bc1=(1-.9**(steps+1)).float();bc2=(1-.999**(steps+1)).sqrt().float()
        outp=[];outm=[];outv=[]
        for p,gg,mm,vv in zip(pp,grads,m,v):
            shape=(-1,)+(1,)*(p.ndim-1);a=(steps<n_steps).float().reshape(shape);l=lr.reshape(shape)
            q=p.detach()*(1-l*.001)
            nm=mm+a*.1*(gg-mm)
            nv=vv+a*((.999*vv+.001*gg*gg)-vv)
            q=q-(l/bc1.reshape(shape))*nm/(nv.sqrt()/bc2.reshape(shape)+1e-8)
            outp.append(q);outm.append(nm);outv.append(nv)
    return outp,outm,outv,[g.detach() for g in grads],logits.detach(),loss.detach(),acc


def metrics(got,ref):
    a=got.detach().double();b=ref.detach().double();d=a-b
    norm=float(torch.linalg.vector_norm(b));err=float(torch.linalg.vector_norm(d))
    return {'max_abs':float(d.abs().max()),'relative_l2':err/norm if norm else (0. if err==0 else None),'reference_l2':norm,'finite':bool(torch.isfinite(a).all()),'pass':bool(torch.isfinite(a).all()) and float(d.abs().max())<=2e-6 and (err<=1e-6*norm if norm else err==0)}


def main():
    output=P/'validation.json';progress=P/'validation_progress.jsonl'
    assert not output.exists() and not progress.exists(),'refuse overwrite'
    result={'complete':False,'pass_all':False,'rows':[],'kernel_calls':0,'torch_reference_calls':0,'kernel_attempts':0,'torch_reference_attempts':0,'interpreter_only':True,'libdevice_shim':['__nv_pow=np.power(float64)','__nv_sqrt=np.sqrt(float64)'],'torch':torch.__version__,'triton':triton.__version__}
    start=time.process_time()
    def save():
        result['cpu_seconds']=time.process_time()-start
        tmp=P/'validation.tmp';tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(output)
    def event(name):
        with progress.open('a') as f:f.write(json.dumps({'event':name,'cpu_seconds':time.process_time()-start})+'\n')
    try:
        gate=json.loads((P/'validation_gate.json').read_text());assert gate['ready']
        mandatory=[str(P/f) for f in ('validate.py','fused_step.py','PROTOCOL.md','manifest.json','trainer_snapshot.py','launch.sh')]
        reviewer_paths=['/workspace/work/recover-small_h2_56/review/fused_rnn_source_review/source_gate.json','/workspace/work/recover-small_h2_910/review/fused_rnn_validation_review/source_gate.json']
        assert all(f in gate['pins'] for f in mandatory+reviewer_paths)
        for key in ('source_publication','kernel_review_publication','validation_review_publication'):
            assert isinstance(gate[key],str) and 8<=len(gate[key])<=40 and all(c in '0123456789abcdef' for c in gate[key])
        for f,h in gate['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
        for path in reviewer_paths:
            review=json.loads(Path(path).read_text());assert review['complete'] and review['pass_all'] and review['execution_allowed']
            required=[str(P/'fused_step.py'),str(P/'PROTOCOL.md'),str(P/'manifest.json')]
            if 'validation_review' in path:required += [str(P/'validate.py'),str(P/'launch.sh')]
            assert all(f in review['pins'] for f in required)
            for f,h in review['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
        limits={'cpu':resource.getrlimit(resource.RLIMIT_CPU),'address_space':resource.getrlimit(resource.RLIMIT_AS)}
        result['limits']=limits;save();assert limits['cpu']==(105,110) and limits['address_space']==(2**31,2**31)
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
        install_libdevice_shim()
        from fused_step import FusedWorkspace
        configs=[('initial',3,32,0,False,False,1),('mixed_active',3,32,5,True,False,1),('odd_batch',2,13,3,True,False,1),('zero_hidden',2,16,1,False,True,1),('sequence',2,17,0,False,False,3),('production_batch',1,1024,7,True,False,1)]
        for index,(name,K,B,step,moments,zero,repeat) in enumerate(configs):
            event('start_'+name);gen=torch.Generator().manual_seed(261000+index)
            params=[torch.randn((K,16),generator=gen)*.4,torch.randn((K,16,16),generator=gen)*.15,torch.randn((K,10,16),generator=gen)*.2]
            if zero:params[0].zero_()
            m=[torch.randn(p.shape,generator=gen)*.01 if moments else torch.zeros_like(p) for p in params]
            v=[torch.rand(p.shape,generator=gen)*.01 if moments else torch.zeros_like(p) for p in params]
            steps=torch.full((K,),float(step),dtype=torch.float64)
            ns=torch.tensor(([3.,6.,12.] if K==3 else [8.,2.] if K==2 else [16.]),dtype=torch.float64)
            warm=torch.arange(K,dtype=torch.float64)+4;lr0=torch.full((K,),.01,dtype=torch.float64)
            workspace=FusedWorkspace(params,B,True);loss=torch.empty(K);acc=torch.empty(K)
            for iteration in range(repeat):
                x=torch.randn((K,B,10),generator=gen);y=x.argsort(-1)[...,-2].contiguous()
                row={'name':name,'iteration':iteration,'K':K,'B':B,'step':steps.tolist(),'status':'started'};result['rows'].append(row);save()
                before=[p.clone() for p in params+m+v];previous=steps.clone()
                assert time.process_time()-start<90
                result['torch_reference_attempts']+=1;save()
                expected=reference(params,m,v,steps,ns,warm,lr0,x,y);result['torch_reference_calls']+=1;save()
                result['kernel_attempts']+=1;save()
                workspace.step(params,m,v,steps,ns,warm,lr0,x,y,loss,acc);result['kernel_calls']+=1;save()
                comparisons={}
                for family,got,ref in [('params',params,expected[0]),('m',m,expected[1]),('v',v,expected[2]),('grad',workspace.grads,expected[3])]:
                    comparisons[family]=[metrics(a,b) for a,b in zip(got,ref)]
                comparisons['logits']=[metrics(workspace.logits,expected[4])];comparisons['loss']=[metrics(loss,expected[5])]
                row['comparisons']=comparisons;row['accuracy_equal']=torch.equal(acc,expected[6]);row['steps_equal']=torch.equal(steps,previous+1)
                inactive=previous>=ns;row['inactive_bitwise']=all(torch.equal(a[inactive],b[inactive]) for a,b in zip(params+m+v,before))
                row['status']='returned';save()
                assert row['accuracy_equal'] and row['steps_equal'] and row['inactive_bitwise']
                assert all(c['pass'] for values in comparisons.values() for c in values),name
                row['status']='passed';save()
            event('complete_'+name)
        assert len(result['rows'])==result['kernel_calls']==result['torch_reference_calls']==result['kernel_attempts']==result['torch_reference_attempts']==8
        assert all(row['status']=='passed' for row in result['rows']) and time.process_time()-start<90
        result['complete']=result['pass_all']=True;save()
    except BaseException:
        result['complete']=result['pass_all']=False;result['failure']=traceback.format_exc();save();raise
if __name__=='__main__':main()
