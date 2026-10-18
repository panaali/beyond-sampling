"""Independent angular integrals for frozen controls and one historical checkpoint."""
import argparse,hashlib,json,math,resource,time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import mpmath as mp
from prefix_mean import term,means,CONST,Counter
HERE=Path(__file__).resolve().parent
protocol=json.loads((HERE/'prefix_protocol.json').read_text())
mp.mp.dps=protocol['reference_precision_digits']
def m(q):
    q=F(q);return mp.mpf(q.numerator)/q.denominator

def angular(a,b):
    a,b=m(a),m(b)
    if not a and not b:return mp.mpf(0)
    lo,hi=-mp.pi/2,mp.pi/2;phi=mp.atan2(b,a);cuts=[lo,hi]
    for k in range(-2,3):
        root=phi+mp.pi/2+k*mp.pi
        if lo<root<hi:cuts.append(root)
    cuts=sorted(cuts);acc=mp.mpf(0)
    for left,right in zip(cuts,cuts[1:]):
        mid=(left+right)/2
        if a*mp.cos(mid)+b*mp.sin(mid)>0:acc+=mp.quad(lambda t:a*mp.cos(t)+b*mp.sin(t),[left,right])
    # Conditional half-normal X: radial expectation sqrt(pi/2), angle density1/pi.
    return mp.sqrt(mp.pi/2)*acc/mp.pi

def controls():
    rows=[]
    for a,b in protocol['control_pairs']:
        for scale in protocol['control_scales']:
            a,b0=float(a)*scale,float(b)*scale;ops=Counter();value=2*CONST*term(a,b0,ops);ref=angular(F(a),F(b0));error=mp.mpf(value)-ref
            passed=abs(error)<=mp.mpf(protocol['absolute_tolerance'])+protocol['relative_tolerance']*abs(ref)
            row=dict(a=a,b=b0,value=value,reference=str(ref),absolute_error=str(abs(error)),passed=bool(passed),naive_value=(math.hypot(a,b0)+a)*2*CONST,counts=dict(ops));rows.append(row);assert passed,row
            # Restore unscaled pair for second scale.
            a=float(a)/scale
    return dict(rows=rows,complete=True,naive_zero_for_positive_reference=sum(r['naive_value']==0 and mp.mpf(r['reference'])>0 for r in rows))

def trained():
    import torch
    path=Path(protocol['weight_path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==protocol['weight_sha256']
    state=torch.load(path,map_location='cpu',weights_only=True);wi=state['rnn.weight_ih_l0'].numpy().astype(float)[:,0];wh=state['rnn.weight_hh_l0'].numpy().astype(float)
    assert len(wi)==16
    start=time.process_time();value,ap,an,ops=means(wi,wh);kernel_cpu=time.process_time()-start
    fw=list(map(F,wi));fwh=[[F(v) for v in row] for row in wh]
    fap=[sum((v*max(w,F(0)) for v,w in zip(row,fw)),F(0)) for row in fwh];fan=[sum((v*max(-w,F(0)) for v,w in zip(row,fw)),F(0)) for row in fwh]
    refs=[(angular(a,b)+angular(c,b))/2 for a,c,b in zip(fap,fan,fw)]
    # Gaussian closure with exact-real preactivation first two moments.
    rows=[]
    for i,(a,c,b,ref,v) in enumerate(zip(fap,fan,fw,refs,value)):
        a,c,b=m(a),m(c),m(b);mu=(a+c)/mp.sqrt(2*mp.pi);var=(a*a+c*c)/2+b*b-mu*mu;sd=mp.sqrt(var)
        gaussian=sd*mp.exp(-mu*mu/(2*var))/mp.sqrt(2*mp.pi)+mu*(1+mp.erf(mu/(sd*mp.sqrt(2))))/2 if sd else max(mu,0)
        err=abs(mp.mpf(float(v))-ref);passed=err<=protocol['absolute_tolerance']+protocol['relative_tolerance']*abs(ref);assert passed,(i,err,ref)
        rows.append(dict(unit=i,wi=float(wi[i]),input_coefficient_nonzero=bool(wi[i]!=0),a_plus_exact=str(fap[i]),a_minus_exact=str(fan[i]),value=float(v),angular_reference=str(ref),absolute_error=str(err),passed=bool(passed),gaussian_mean=str(gaussian),preactivation_mean=str(mu),preactivation_sd=str(sd)))
    scale=mp.sqrt(sum(v*v for v in refs)/len(refs));relative=mp.sqrt(sum((mp.mpf(r['gaussian_mean'])-ref)**2 for r,ref in zip(rows,refs))/len(refs))/scale
    return dict(complete=True,weight_sha256=protocol['weight_sha256'],rows=rows,all_input_coefficients_nonzero=bool(np.all(wi!=0)),all_exact_means_positive=all(r>0 for r in refs),gaussian_relative_rms_mean_error=str(relative),kernel_counts=ops,kernel_named_total=sum(ops.values()),kernel_cpu_seconds=kernel_cpu,reference_cost_charged_separately=True,no_model_forward_passes=True,no_random_inputs=True,no_final_accuracy=True,qualification='Floating angle/elementary evaluation comparison, not an outward arithmetic certificate; exact-real formula independent of finite Monte Carlo.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('controls','trained'));args=parser.parse_args();resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time();out=globals()[args.mode]();out.update(cpu_seconds=time.process_time()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,mode=args.mode)
    (HERE/('prefix_'+args.mode+'.json')).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('rows',)}))
if __name__=='__main__':main()
