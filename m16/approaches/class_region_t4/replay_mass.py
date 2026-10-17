"""Refine recorded Plackett mass calls without repeating or modifying geometry."""
import argparse,hashlib,json,resource,time
import numpy as np
from . import runner as R

def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['quadratic','nearest']);p.add_argument('multiplier',type=int,choices=[2,4]);a=p.parse_args();resource.setrlimit(resource.RLIMIT_CPU,(55,57));start=time.process_time()
    prior=json.loads((R.HERE/(a.kind+'_exact.json')).read_text());path=R.HERE/prior['mass_capture']['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==prior['mass_capture']['sha256']
    data=np.load(path);rays=data['rays'];old=data['values'];assert prior['complete'] and abs(old.sum()-prior['partial_completed_error'])<1e-12
    out=dict(kind=a.kind,multiplier=a.multiplier,complete=False,source_capture_sha256=prior['mass_capture']['sha256'],total_mass_calls=len(rays),old_error_sum=float(old.sum()),old_estimate=prior['estimate'],old_signed_error=prior['signed_reference_error'],geometry_changed=False,portable_cost_certified=False)
    ops=R.Ops(50);values=[];target=R.HERE/(a.kind+'_mass_x'+str(a.multiplier)+'.json')
    def save():
        out.update(completed_mass_calls=len(values),partial_error_sum=float(sum(values)),partial_named_ops=ops.n,cpu=time.process_time()-start)
        target.write_text(json.dumps(out,indent=2)+'\n')
    try:
        R.engine._RULES={'wide':R.engine._rule(12*a.multiplier,2),'thin':R.engine._rule(48*a.multiplier,3)}
        R.engine.STATS.update({k:0 for k in R.engine.STATS})
        for i,r in enumerate(rays):
            values.append(R.engine.orthant(r,ops))
            if (i+1)%128==0:save()
        values=np.array(values);estimate=1-float(values.sum())
        out.update(complete=True,estimate=estimate,signed_reference_error=estimate-R.REFERENCE,default_minus_refined=float(prior['estimate']-estimate),negative_values=int(sum(values<0)),zero_values=int(sum(values==0)),max_per_simplex_change=float(np.max(abs(values-old))),maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        np.savez_compressed(R.HERE/(a.kind+'_mass_x'+str(a.multiplier)+'_values.npz'),values=values)
    except Exception as exc:out.update(failure_type=type(exc).__name__,failure=str(exc),partial_accuracy_not_reported=True)
    save();print(json.dumps(out,indent=2))
if __name__=='__main__':main()
