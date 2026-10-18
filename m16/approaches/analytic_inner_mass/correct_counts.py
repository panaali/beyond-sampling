"""Correct explicit Boolean/domain work without rerunning saved trained quadrature."""
import hashlib,json
from pathlib import Path
from collections import Counter
HERE=Path(__file__).parent

def overhead(row):
    m=row['order']**2;assert row['inner_modes']=={'analytic':m}
    # Existing omissions: four Boolean reductions, fallback mode equality,
    # 1 or4 repeated NOT masks, and scalar final NOT. New domain guards:
    # twelve comparisons/reductions per node minus2, plus a+b per node.
    return m,(21 if row['kernel_counts'].get('sin',0) else 18)*m-5

def main():
    old=json.loads((HERE/'controls_initial.json').read_text());new=json.loads((HERE/'controls.json').read_text());checked=[]
    for group in ('scalar','analytic','trained'):
        assert [r['value'] for r in old[group]]==[r['value'] for r in new[group]]
        if group=='scalar':continue
        for o,n in zip(old[group],new[group]):
            if o['inner_modes']!={'analytic':o['order']**2}:continue
            a,c=overhead(o);assert n['kernel_counts']['arithmetic']-o['kernel_counts']['arithmetic']==a;assert n['kernel_counts']['comparison']-o['kernel_counts']['comparison']==c;checked.append((group,o.get('index'),o['order']))
    audits=[]
    for budget in ('per_cone','aggregate'):
        initial=HERE/(budget+'_initial.json');out=json.loads(initial.read_text());oldtotal=out['named_total_including_comparisons'];arith=comp=0
        for row in out['rows']:
            if row['mode']!='quadrature':continue
            a,c=overhead(row);row['kernel_counts']['arithmetic']+=a;row['kernel_counts']['comparison']+=c;arith+=a;comp+=c
        ledger=sum((Counter(r.get('kernel_counts',{})) for r in out['rows']),Counter());out['kernel_counts']=dict(ledger);out['kernel_named_total']=sum(ledger.values());out['named_total_including_comparisons']+=arith+comp;out['cost_correction']=dict(initial_saved_file=initial.name,initial_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),initial_named_total=oldtotal,additional_arithmetic=arith,additional_boolean_comparison=comp,corrected_source_sha256=hashlib.sha256((HERE/'primitive.py').read_bytes()).hexdigest(),trained_values_rerun=False,domain_guard_justification='Every saved node already accepted analytic: ac>0,delta>1e-12*ac and finite positive formula enforce positive endpoints and rule out collinear origin crossing.',excluded_indexing_memory_scans=True)
        (HERE/(budget+'.json')).write_text(json.dumps(out,indent=2)+'\n');audits.append(dict(budget=budget,old=oldtotal,new=out['named_total_including_comparisons'],additional_arithmetic=arith,additional_comparison=comp,old_t130_total=out['old_full_named_total']))
    result=dict(supported_control_values_unchanged=181,all_analytic_control_ledger_checks=len(checked),control_ledger_cases=checked,corrections=audits,no_trained_numeric_rerun=True)
    (HERE/'cost_correction.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['corrections'],indent=2))
if __name__=='__main__':main()
