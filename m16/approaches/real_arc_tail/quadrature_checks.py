from pathlib import Path
import json,time
import numpy as np
import mpmath as mp
from recurrence import endpoint,baseline
from controls import polygons
from quadrature import ArcRule
from pilot import load_tail

def main():
    start=time.process_time();mp.mp.dps=100
    rule=ArcRule();arcrows=[]
    for row in json.loads(Path(__file__).with_name('reference_repair_results.json').read_text())['cases']:
        vals,bounds=rule.evaluate(row['a'],row['b'],row['theta'],row['degree'])
        ref=mp.mpf(row['reference']);err=abs(mp.mpf(vals[-1])-ref)
        arcrows.append({k:row[k] for k in ('a','b','theta','degree','reference','true_float_underflow')}|dict(value=vals[-1],integration_bound=bounds[-1],relative=float(err/ref) if ref else 0.,absolute_error=float(err)))
    prows=[]
    for i,vs in enumerate(polygons()):
        for k in (2,4,6,8):
            stats={};mod=rule.module();r={'case':i,'k':k,'endpoint_moment':endpoint.polygon_moment(vs,k-1),'quadrature_moment':mod.polygon_moment(vs,k-1,stats)}
            try:r['guarded']=mod.tail_mass(vs,k,stats)
            except ArithmeticError as e:r['guard_failure']=str(e);r['fallback']=mod.tail_mass_with_fallback(vs,k,stats)
            r['stats']=stats;prows.append(r)
    import torch
    rows=[]
    saved=json.loads(Path(__file__).with_name('pilot_results.json').read_text())
    old=load_tail('quadrature_baseline',baseline.tail_mass_with_fallback)
    for prior in saved['cases']:
        T=prior['T'];t=time.process_time()
        path=Path(f'/workspace/clean-neurons/work/checkpoints/H2_T{T}_seed0.pth');state=torch.load(path,map_location='cpu',weights_only=True)
        model=tuple(state[k].numpy().astype(float).reshape(-1) if k=='rnn.weight_ih_l0' else state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
        h,top=old.prefix_summary(model,np.array(prior['prefix']));setup=time.process_time()-t
        r={'T':T,'sign':prior['sign'],'setup_cpu':setup,'historical_endpoint':prior['endpoint']['probability']}
        for name in ('endpoint','quadrature'):
            repetitions=[]
            for repeat in range(4):
                t=time.process_time()
                if name=='quadrature':
                    qr=ArcRule();mod=load_tail('bounded_tail',qr.module().tail_mass_with_fallback)
                else:mod=load_tail('matched_endpoint',endpoint.tail_mass_with_fallback)
                value,stats=mod.conditional_probability(model,h,top,T-2)
                if name=='quadrature':stats['rule_generations']=1;stats['rule_nodes']=8;stats['rule_generation_cpu']=qr.generation_cpu
                repetitions.append(dict(value=value,cpu=time.process_time()-t,stats=stats))
            r[name]=repetitions
        rows.append(r)
    out=dict(arcs=arcrows,polygons=prows,cases=rows,cpu_seconds=time.process_time()-start,initial_rule_generation_cpu=rule.generation_cpu,integration_bound_excludes_roundoff=True)
    Path(__file__).with_name('quadrature_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(cpu_seconds=out['cpu_seconds'],max_arc_relative_representable=max(r['relative'] for r in arcrows if not r['true_float_underflow']),polygon_fallbacks=sum('guard_failure'in r for r in prows),max_tail_difference=max(abs(r['quadrature'][0]['value']-r['endpoint'][0]['value']) for r in rows),max_tail_integration_bound=max(r['quadrature'][0]['stats']['tail_arc_integration_error_sum'] for r in rows),summed_query_cpu={m:sum(v['cpu'] for r in rows for v in r[m]) for m in ('endpoint','quadrature')}),indent=2))
if __name__=='__main__':main()
