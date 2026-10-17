"""Frozen finite bound-driven rules on exactly56 remaining original cones."""
import hashlib,json,math,time,resource
from pathlib import Path
import numpy as np
import mpmath as mp
from .radial import bounds,probability_bound,integrate
HERE=Path(__file__).parent

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time();assert json.loads((HERE/'controls.json').read_text())['all_pass'];path=HERE.parent/'class_region_t4/quadratic_exact_mass_calls.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()=='07a0720e229bd2e0aeda511b60038efdc37f76c17aadfdeb17cf0caf3f264581'
    rays=np.load(path)['rays'];ids=json.loads((HERE.parent/'acute_t4_reference/results.json').read_text())['uncovered'];assert len(ids)==56;source=np.load(HERE.parent/'cone_mass4_review/quadratic_values.npz')['values'];out=dict(rows=[],complete=False,full_float_accuracy_certified=False,portable_cost_certified=False);values=np.full(len(rays),np.nan);bvals=np.full(len(rays),np.nan)
    def save():
        ok=np.isfinite(values);out.update(cpu=time.process_time()-start,completed=int(ok.sum()),reference_mass=math.fsum(values[ok]),integration_only_bound=math.fsum(bvals[ok]),source_minus_positive=math.fsum(source[ok]-values[ok]),known_integrand_arithmetic=sum(r.get('known_integrand_arithmetic',0) for r in out['rows']))
        (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n');np.savez_compressed(HERE/'radial_values.npz',values=values,bounds=bvals)
    for i in ids:
        if time.process_time()-start>18:out['stop_reason']='18CPU soft cap within20CPU review lane';break
        row=dict(index=i,source=float(source[i]));out['rows'].append(row)
        try:
            t=time.process_time();cert=bounds(rays[i]);row.update(certificate=cert,exact_setup_cpu=time.process_time()-t,trials=[]);t=time.process_time()
            for n in (20,24,28,32,40,48,64):
                bd=probability_bound(cert,n);row['trials'].append(dict(order=n,**bd))
                if bd['bound']<=1e-9/56:break
            row.update(order=n,target_pass=bd['bound']<=1e-9/56,bound_cpu=time.process_time()-t);t=time.process_time();values[i]=integrate(rays[i],n);bvals[i]=bd['bound'];row.update(value=values[i],integration_only_bound=bvals[i],difference=float(source[i]-values[i]),quadrature_cpu=time.process_time()-t,known_integrand_arithmetic=39*n**3)
        except Exception as exc:row['failure']=type(exc).__name__+': '+str(exc)
        if len(out['rows'])%8==0:save()
    else:out['complete']=True
    save();print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
    if out['complete'] and np.isfinite(values[ids]).all():
        pos=np.load(HERE.parent/'positive_t4_corpus/positive_values.npz');acute=np.load(HERE.parent/'acute_t4_reference/acute_values.npz');allv=values.copy();allb=bvals.copy()
        for other in (pos,acute):
            mask=np.isfinite(other['values']);assert not np.any(np.isfinite(allv[mask]));allv[mask]=other['values'][mask];allb[mask]=other['bounds'][mask]
        assert np.isfinite(allv).all();total=dict(cones=len(allv),estimate=1-math.fsum(allv),integration_only_bound=math.fsum(allb),reference=.477013125994119,reference_difference=1-math.fsum(allv)-.477013125994119,negative_count=int(sum(allv<0)),full_float_accuracy_certified=False,portable_cost_certified=False)
        (HERE/'combined.json').write_text(json.dumps(total,indent=2)+'\n');np.savez_compressed(HERE/'combined_values.npz',values=allv,bounds=allb);print(json.dumps(total,indent=2))
if __name__=='__main__':main()
