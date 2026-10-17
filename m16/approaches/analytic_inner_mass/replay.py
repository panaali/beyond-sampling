"""Unchanged prior T130 selected nodes/tolerances; only replace integration kernel."""
import argparse,hashlib,json,math,resource,time
from pathlib import Path
from collections import Counter
import numpy as np
from .primitive import Rules,integrate
HERE=Path(__file__).parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('budget',choices=('per_cone','aggregate'));args=parser.parse_args();resource.setrlimit(resource.RLIMIT_CPU,(110,110));start=time.process_time();assert not (HERE/(args.budget+'.json')).exists();controls=json.loads((HERE/'controls.json').read_text());assert all(controls[k] for k in ('scalar_all_pass','analytic_all_pass','trained_all_pass'))
    raypath=HERE.parent/'class_region_t4/quadratic_exact_mass_calls.npz';assert hashlib.sha256(raypath.read_bytes()).hexdigest()=='07a0720e229bd2e0aeda511b60038efdc37f76c17aadfdeb17cf0caf3f264581';rays=np.load(raypath)['rays'];basepath=HERE.parent/'positive_mass_budget'/(args.budget+'.json');base=json.loads(basepath.read_text());assert base['complete'] and len(base['rows'])==6347;refpath=HERE.parent/'radial_t4_reference/combined_values.npz';reference=np.load(refpath);rules=Rules();values=np.full(6347,np.nan);series=np.zeros(6347);bds=np.full(6347,np.nan);out=dict(budget=args.budget,base_sha256=hashlib.sha256(basepath.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),source_sha256=hashlib.sha256((HERE/'primitive.py').read_bytes()).hexdigest(),rows=[],complete=False,changed_geometry=False,changed_order_selection=False,replay_not_new_end_to_end_timing=True,full_float_certificate=False,complete_implementation_cost=False)
    def save():
        ok=np.isfinite(values);ledger=sum((Counter(r.get('kernel_counts',{})) for r in out['rows']),Counter());modes=sum((Counter(r.get('inner_modes',{})) for r in out['rows']),Counter());setup=sum(sum(r['setup_selection_counts'].values()) for r in base['rows'][:len(out['rows'])]);constant=sum(2*('interval' in r)+2*bool(r['trials'])+2*(r['mode']=='quadrature') for r in base['rows'][:len(out['rows'])]);kernel=sum(ledger.values());node=sum(rules.ledger.counts.values());oldkernel=sum(r['kernel_arithmetic'] for r in base['rows'][:len(out['rows'])]);named=setup+constant+kernel+node
        out.update(cpu=time.process_time()-start,attempted=len(out['rows']),covered=int(ok.sum()),unresolved_indices=[r['index'] for r in out['rows'] if r['mode']=='unresolved'],covered_error=math.fsum(values[ok]-reference['values'][ok]),covered_absolute_error=math.fsum(abs(values[ok]-reference['values'][ok])),max_absolute_error=float(max(abs(values[ok]-reference['values'][ok]),default=0)),bound_sum=math.fsum(bds[ok]),series_error_bound_sum=math.fsum(series[ok]),bound_violations=int(sum(abs(values[ok]-reference['values'][ok])>bds[ok]+series[ok]+reference['bounds'][ok]+5e-14)),kernel_counts=dict(ledger),inner_modes=dict(modes),kernel_named_total=kernel,old_same_rows_kernel_arithmetic=oldkernel,cold_node_plan_counts=rules.ledger.dict(),inherited_setup_selection_count=setup,inherited_ordinary_constant_arithmetic=constant,named_total_including_comparisons=named,old_full_named_total=base['named_total_including_comparisons'],cold_rules=rules.events)
        if ok.all():out.update(estimate=1-math.fsum(values),reference_error=1-math.fsum(values)-.477013125994119)
        (HERE/(args.budget+'.json')).write_text(json.dumps(out,indent=2,default=lambda x:x.item())+'\n');np.savez_compressed(HERE/(args.budget+'_values.npz'),values=values,bounds=bds,series_bounds=series)
    for i,previous in enumerate(base['rows']):
        if time.process_time()-start>90:out['stop_reason']='90CPU soft cap';break
        assert previous['index']==i;row=dict(index=i,mode=previous['mode'],old_value=previous.get('value'),old_bound=previous.get('bound'))
        try:
            if previous['mode']=='quadrature':
                row.update(integrate(rays[i],previous['order'],previous['det'],rules));series[i]=row['series_error_bound'];row['difference_from_prior_kernel']=row['value']-previous['value']
            elif 'value' in previous:row['value']=previous['value']
            if 'value' in row:values[i]=row['value'];bds[i]=previous['bound']
        except Exception as exc:row['failure']=type(exc).__name__+': '+str(exc);out['stop_reason']='Kernel failure preserved';out['rows'].append(row);break
        out['rows'].append(row)
        if (i+1)%256==0:save()
    else:out['complete']=True
    save();print(json.dumps({k:v for k,v in out.items() if k not in ('rows','cold_rules','unresolved_indices')},indent=2));assert out['complete']
if __name__=='__main__':main()
