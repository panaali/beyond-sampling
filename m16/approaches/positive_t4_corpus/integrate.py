"""Positive references for represented cones only; explicit uncovered residual."""
import json,math,time,resource
from pathlib import Path
import numpy as np
from ..cone_mass4_review.check_controls import positive_mass
HERE=Path(__file__).parent

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(110,112));start=time.process_time();R=np.load(HERE/'represented_rays.npz')['rays'];inv=json.loads((HERE/'inventory.json').read_text());parent=HERE.parent/'cone_mass4_review';source=np.load(parent/'quadratic_values.npz')['values'];guard=np.load(parent/'quadratic_rotation_values.npz')['values'];rows=[];values=np.full(len(R),np.nan);bounds=np.full(len(R),np.nan);out=dict(complete=False,full_accuracy_certified=False,full_cost_certified=False,uncovered=inv['uncovered'])
    def save():
        done=np.isfinite(values);covered=np.flatnonzero(done);uncovered=np.array(inv['uncovered']);d=source[done]-values[done];out.update(completed=int(done.sum()),represented=5859,cpu=time.process_time()-start,reference_sum=math.fsum(values[done]),integration_only_bound_sum=math.fsum(bounds[done]),source_covered_sum=math.fsum(source[done]),guard_covered_sum=math.fsum(guard[done]),source_covered_difference=math.fsum(d),source_absolute_difference_sum=math.fsum(abs(d)),max_absolute_difference=float(max(abs(d),default=0)),uncovered_source_sum=math.fsum(source[uncovered]),uncovered_guard_sum=math.fsum(guard[uncovered]),known_integrand_arithmetic=int(done.sum())*312000,rows=rows)
        if len(covered):out['largest_discrepancies']=sorted([dict(index=int(i),reference=float(values[i]),source=float(source[i]),difference=float(source[i]-values[i]),bound=float(bounds[i])) for i in covered],key=lambda r:-abs(r['difference']))[:20]
        (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n');np.savez_compressed(HERE/'positive_values.npz',values=values,bounds=bounds)
    for i,Y in enumerate(R):
        if not np.isfinite(Y).all():continue
        if time.process_time()-start>90:out['stop_reason']='predeclared90CPU soft cap';break
        try:
            values[i],bounds[i]=positive_mass(Y,20);Z=Y/Y.sum(1)[:,None];rows.append(dict(index=i,normalized_det=float(np.linalg.det(Z)),condition=float(np.linalg.cond(Z))))
        except Exception as exc:out.setdefault('failures',[]).append(dict(index=i,type=type(exc).__name__,message=str(exc)))
        if len(rows)%128==0:save()
    else:out['complete']=True
    save();print(json.dumps({k:v for k,v in out.items() if k not in ('rows','uncovered','largest_discrepancies')},indent=2))
if __name__=='__main__':main()
