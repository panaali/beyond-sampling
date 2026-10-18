"""Two frozen objectives, separate cold runs; preserve incomplete coverage."""
import argparse,hashlib,json,math,time,resource
from pathlib import Path
from collections import Counter
import numpy as np
from .primitive import Rules,estimate
HERE=Path(__file__).parent

def main():
    p=argparse.ArgumentParser();p.add_argument('budget',choices=('per_cone','aggregate'));args=p.parse_args();resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time();assert json.loads((HERE/'controls.json').read_text())['all_returned_analytic_pass'];path=HERE.parent/'class_region_t4/quadratic_exact_mass_calls.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()=='07a0720e229bd2e0aeda511b60038efdc37f76c17aadfdeb17cf0caf3f264581';rays=np.load(path)['rays'];ref=np.load(HERE.parent/'radial_t4_reference/combined_values.npz');tol=1e-7 if args.budget=='per_cone' else 1e-7/6347;rules=Rules();out=dict(budget=args.budget,tolerance=tol,rows=[],complete=False,full_float_accuracy_certified=False,portable_cost_certified=False);values=np.full(len(rays),np.nan);bounds=np.full(len(rays),np.nan)
    def save():
        ok=np.isfinite(values);counts={k:sum(r['setup_selection_counts'][k] for r in out['rows']) for k in ('arithmetic','comparison','sqrt','abs')};kernel=sum(r['kernel_arithmetic'] for r in out['rows']);constant_arithmetic=sum(2*('interval' in r)+2*bool(r['trials'])+2*(r['mode']=='quadrature') for r in out['rows']);named=sum(counts.values())+kernel+rules.node_plan_arithmetic+constant_arithmetic
        out.update(cpu=time.process_time()-start,attempted=len(out['rows']),covered=int(ok.sum()),modes=dict(Counter(r['mode'] for r in out['rows'])),orders=dict(Counter(str(r.get('order')) for r in out['rows'] if 'order' in r)),covered_error=math.fsum(values[ok]-ref['values'][ok]),covered_absolute_error=math.fsum(abs(values[ok]-ref['values'][ok])),bound_sum=math.fsum(bounds[ok]),max_absolute_error=float(max(abs(values[ok]-ref['values'][ok]),default=0)),bound_violations=int(sum(abs(values[ok]-ref['values'][ok])>bounds[ok]+ref['bounds'][ok]+5e-14)),counts=counts,ordinary_pi_constant_arithmetic=constant_arithmetic,kernel_arithmetic=kernel,cold_node_plan_arithmetic=rules.node_plan_arithmetic,named_total_including_comparisons=named,named_average_per_attempt=named/len(out['rows']) if out['rows'] else None,cold_rules=rules.events)
        if ok.all():out.update(estimate=1-math.fsum(values),reference_error=1-math.fsum(values)-.477013125994119)
        (HERE/(args.budget+'.json')).write_text(json.dumps(out,indent=2)+'\n');np.savez_compressed(HERE/(args.budget+'_values.npz'),values=values,bounds=bounds)
    for i,R in enumerate(rays):
        if time.process_time()-start>18:out['stop_reason']='18CPU soft cap';break
        row=estimate(R,tol,rules);row['index']=i;out['rows'].append(row)
        if 'value' in row:values[i]=row['value'];bounds[i]=row['bound']
        if (i+1)%128==0:save()
    else:out['complete']=True
    save();print(json.dumps({k:v for k,v in out.items() if k not in ('rows','cold_rules')},indent=2))
if __name__=='__main__':main()
