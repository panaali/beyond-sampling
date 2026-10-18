"""T261: one Triton launch for forward/BPTT/AdamW on supplied x and labels.

H=16,T=10, float32 parameters/data; float64 schedule. One CTA per model.
Data sampling and second-argmax labels intentionally retain Trainer's source outside
this kernel. CPU interpreter is a correctness tool, not a GPU performance proxy.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _adam(P,M,V,offset,mask,g,lr,bc1,bc2_sqrt,active,WD:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,EPS:tl.constexpr):
    p=tl.load(P+offset,mask,0);m=tl.load(M+offset,mask,0);v=tl.load(V+offset,mask,0)
    # Retain original expression order. No FMA contraction at launch.
    act=active.to(tl.float32)
    mn=m+act*(1.0-B1)*(g-m)
    vn=v+act*((B2*v+(1.0-B2)*g*g)-v)
    denom=tl.sqrt(vn)/bc2_sqrt+EPS
    pn=p*(1.0-lr*WD)-(lr/bc1)*mn/denom
    tl.store(P+offset,pn,mask);tl.store(M+offset,mn,mask);tl.store(V+offset,vn,mask)


@triton.jit
def _step(X,Y,WI,WH,WO,MI,MH,MO,VI,VH,VO,STEPS,NSTEPS,WARM,LR0,
          HS,LOSS,ACC,GI,GH,GO,LOGITS,
          B:tl.constexpr,BB:tl.constexpr,T:tl.constexpr,
          FLOOR:tl.constexpr,WD:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,EPS:tl.constexpr,
          DIAGNOSTICS:tl.constexpr):
    k=tl.program_id(0); b=tl.arange(0,BB); j=tl.arange(0,16)
    wi=tl.load(WI+k*16+j)
    wh=tl.load(WH+k*256+j[:,None]*16+j[None,:])
    wo=tl.load(WO+k*T*16+j[:,None]*16+j[None,:],j[:,None]<T,0)
    h=tl.full((BB,16),0,tl.float32)
    tl.store(HS+k*(T+1)*B*16+b[:,None]*16+j[None,:],h,b[:,None]<B)
    for t in range(T):
        x=tl.load(X+k*B*T+b*T+t,b<B,0)
        h=tl.maximum(tl.dot(h,tl.trans(wh),input_precision='ieee')+x[:,None]*wi[None,:],0)
        tl.store(HS+k*(T+1)*B*16+(t+1)*B*16+b[:,None]*16+j[None,:],h,b[:,None]<B)
    logits=tl.dot(h,tl.trans(wo),input_precision='ieee')
    logits=tl.where(j[None,:]<T,logits,float('-inf'))
    maximum=tl.max(logits,1)
    exponent=tl.exp(logits-maximum[:,None]);den=tl.sum(exponent,1)
    prob=exponent/den[:,None]
    y=tl.load(Y+k*B+b,b<B,0)
    truth_logit=tl.sum(tl.where(j[None,:]==y[:,None],logits,0),1)
    losses=maximum+tl.log(den)-truth_logit
    pred=tl.min(tl.where(logits==maximum[:,None],j[None,:],2147483647),1)
    tl.store(LOSS+k,tl.sum(tl.where(b<B,losses,0),0)/B)
    tl.store(ACC+k,tl.sum(((pred==y)&(b<B)).to(tl.float32),0)/B)
    dl=tl.where((b[:,None]<B)&(j[None,:]<T),(prob-(j[None,:]==y[:,None]).to(tl.float32))/B,0)
    gw_o=tl.dot(tl.trans(dl),h,input_precision='ieee')
    dh=tl.dot(dl,wo,input_precision='ieee')
    gw_h=tl.full((16,16),0,tl.float32);gw_i=tl.full((16,),0,tl.float32)
    # All within-CTA forward scratch writes must be visible to backward layouts.
    tl.debug_barrier()
    for r in range(T):
        t=T-1-r
        curr=tl.load(HS+k*(T+1)*B*16+(t+1)*B*16+b[:,None]*16+j[None,:],b[:,None]<B,0)
        prev=tl.load(HS+k*(T+1)*B*16+t*B*16+b[:,None]*16+j[None,:],b[:,None]<B,0)
        dz=tl.where(curr>0,dh,0)
        x=tl.load(X+k*B*T+b*T+t,b<B,0)
        gw_i+=tl.sum(dz*x[:,None],0)
        gw_h+=tl.dot(tl.trans(dz),prev,input_precision='ieee')
        dh=tl.dot(dz,wh,input_precision='ieee')
    step=tl.load(STEPS+k);n=tl.load(NSTEPS+k);warm=tl.load(WARM+k);lr0=tl.load(LR0+k)
    active=step<n
    floor64=tl.full((),FLOOR,tl.float64)
    decay=tl.maximum(libdevice.pow(floor64,2.0*step/n),floor64)
    mult=tl.minimum(step/warm,1.0)*decay
    lr=(lr0*mult*active.to(tl.float64)).to(tl.float32)
    one=tl.full((),1.0,tl.float64)
    bc1=(one-libdevice.pow(tl.full((),B1,tl.float64),step+one)).to(tl.float32)
    bc2_sqrt=libdevice.sqrt(one-libdevice.pow(tl.full((),B2,tl.float64),step+one)).to(tl.float32)
    if DIAGNOSTICS:
        tl.store(GI+k*16+j,gw_i)
        tl.store(GH+k*256+j[:,None]*16+j[None,:],gw_h)
        tl.store(GO+k*T*16+j[:,None]*16+j[None,:],gw_o,j[:,None]<T)
        tl.store(LOGITS+k*B*T+b[:,None]*T+j[None,:],logits,(b[:,None]<B)&(j[None,:]<T))
    _adam(WI,MI,VI,k*16+j,tl.full((16,),True,tl.int1),gw_i,lr,bc1,bc2_sqrt,active,WD,B1,B2,EPS)
    _adam(WH,MH,VH,k*256+j[:,None]*16+j[None,:],tl.full((16,16),True,tl.int1),gw_h,lr,bc1,bc2_sqrt,active,WD,B1,B2,EPS)
    _adam(WO,MO,VO,k*T*16+j[:,None]*16+j[None,:],j[:,None]<T,gw_o,lr,bc1,bc2_sqrt,active,WD,B1,B2,EPS)
    # Per-model counters: no program reads another program's mutable scalar.
    tl.store(STEPS+k,step+one)


class FusedWorkspace:
    def __init__(self,params,batch,diagnostics=False):
        wi,wh,wo=params;K,H=wi.shape;T=wo.shape[1]
        if H!=16 or T!=10 or not 1<=batch<=1024:raise ValueError('requires H16,T10,1<=B<=1024')
        if any(p.dtype!=torch.float32 or not p.is_contiguous() or p.device!=wi.device for p in params):raise ValueError('contiguous same-device float32 parameters required')
        if wh.shape!=(K,16,16) or wo.shape!=(K,10,16):raise ValueError('parameter shapes')
        self.K,self.B,self.T=K,batch,T;self.diagnostics=diagnostics
        self.hidden=torch.empty((K,T+1,batch,H),device=wi.device,dtype=wi.dtype)
        self.grads=[torch.empty_like(p) for p in params]
        self.logits=torch.empty((K,batch,T),device=wi.device,dtype=wi.dtype)

    def step(self,params,m,v,steps,n_steps,warm,lr0,x,y,loss,acc,wd=1e-3,b1=.9,b2=.999,eps=1e-8,floor=.01):
        if x.shape!=(self.K,self.B,self.T) or y.shape!=(self.K,self.B):raise ValueError('batch shapes')
        if x.dtype!=torch.float32 or y.dtype!=torch.int64 or not x.is_contiguous() or not y.is_contiguous():raise ValueError('contiguous fp32 x/int64 y required')
        if any(z.shape!=(self.K,) or z.dtype!=torch.float64 or not z.is_contiguous() for z in (steps,n_steps,warm,lr0)):raise ValueError('per-model float64 schedule vectors required')
        _step[(self.K,)](x,y,*params,*m,*v,steps,n_steps,warm,lr0,self.hidden,loss,acc,*self.grads,self.logits,
                         self.B,triton.next_power_of_2(max(16,self.B)),self.T,floor,wd,b1,b2,eps,self.diagnostics,
                         num_warps=8 if self.B<=256 else 16,enable_fp_fusion=False)


def trainer_class(base_module):
    """Return a Trainer-compatible class. Instantiate instead of base_module.Trainer.

    Input generation/argsort are exactly the inherited source and are outside the
    single compute/update kernel. CUDA graph capture retains them as before.
    Checkpoints export the original scalar step; restoring broadcasts it to all K
    counters. _capture resets the full vector after its warmup, avoiding races.
    """
    class FusedTrainer(base_module.Trainer):
        def __init__(self,*args,**kwargs):
            graph=kwargs.pop('graph',None)
            super().__init__(*args,graph=False,**kwargs)
            if self.H!=16 or self.T!=10:raise ValueError('fused specialization H16,T10 only')
            self.step_t=self.step_t.expand(self.K).clone()
            self.warm=self.warm.contiguous()
            self.n_steps=self.n_steps.contiguous()
            self.lr0=self.lr0.contiguous()
            self.fused=FusedWorkspace(self.params,self.B)
            if graph is None:graph=self.device!='cpu'
            if graph:self._capture()
        def _step_body(self):
            x=base_module.sample(self.K,self.B,self.T,self.gen,self.device,*self.mix)
            y=x.argsort(-1)[...,-2].contiguous()
            with torch.no_grad():
                self.fused.step(self.params,self.m,self.v,self.step_t,self.n_steps,self.warm,self.lr0,x,y,self.loss,self.acc,
                                wd=self.wd,b1=self.b1,b2=self.b2,eps=self.eps,floor=self.floor)
        def state(self):
            if not bool(torch.all(self.step_t==self.step_t[0]).item()):raise RuntimeError('divergent per-model steps')
            st=super().state();st['step']=self.step_t[0].detach().cpu().clone();return st
    return FusedTrainer
