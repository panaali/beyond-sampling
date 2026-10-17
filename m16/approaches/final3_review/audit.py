"""Independent raw E45 arithmetic, simultaneous truth sensitivity, finite trends."""
from pathlib import Path
import json,math,time
import numpy as np
from scipy.stats import beta
HERE=Path(__file__).parent

def close(a,b):return abs(a-b)<=5e-12*max(1.,abs(a),abs(b))
def ratio(estimate,p,n):return n*(estimate-p)**2/(p*(1-p))
def main():
    stamp=time.process_time();read=lambda n:json.loads((HERE/'snapshot'/n).read_text());raw=read('res_H8_T4.json');exact=read('exact_H8_T4.json');summary=read('summary.json');nomass=read('cells_nomass.json');T3=read('evalH_8.json')['zoo'];M=4_000_000;alpha=.05/len(raw);rows=[]
    fpe=6000*(2*4*8*8+4*4*8)/(2*10*16*16+4*10*16);assert fpe==2000/3
    for name,r in raw.items():
        p=r['truth'];estimate=r['est'];n=r['ops']/fpe;smse=p*(1-p)/n;rawratio=ratio(estimate,p,n);corrected=max((estimate-p)**2-p*(1-p)/M,0)/smse
        checks={'N_equiv':close(n,r['N_equiv']),'err_pp':close(100*(estimate-p),r['err_pp']),'ratio_raw':close(rawratio,r['ratio_raw']),'ratio_corrected':close(corrected,r['ratio']),'summary_copy':all(summary['T4'][name][k]==v for k,v in r.items())};assert all(checks.values()),(name,checks)
        successes=round(p*M);assert abs(successes-p*M)<1e-7
        lo=float(beta.ppf(alpha/2,successes,M-successes+1));hi=float(beta.ppf(1-alpha/2,successes+1,M-successes))
        candidates=[ratio(estimate,lo,n),ratio(estimate,hi,n)]+([0.] if lo<=estimate<=hi else [])
        row=dict(name=name,checks=checks,truth_successes=successes,nominal_fpe=fpe,nominal_equivalents=n,raw_ratio=rawratio,corrected_ratio=corrected,truth_simultaneous95_interval=[lo,hi],ratio_over_interval=[min(candidates),max(candidates)],estimate_below_interval=estimate<lo)
        if name in exact:
            e=exact[name];gap=(estimate-e['acc'])**2/smse
            assert close(gap,summary['T4'][name]['gap_ratio']);assert close(100*(e['acc']-p),e['exact_minus_mc_pp']);assert close(100*(estimate-e['acc']),e['recipe_minus_exact_pp'])
            row.update(recipe_minus_observed_exact=estimate-e['acc'],gap_ratio=gap,recipe_to_exact_cost=r['ops']/e['ops'],exact_cells=e['cells'])
        rows.append(row)
    clean=[r for k,r in raw.items() if 'noisy' not in k];mean=lambda xs:sum(xs)/len(xs);c3=mean([r['clean_cells'] for r in T3.values()]);r3exact=mean([r['regions'] for r in T3.values()]);r3lb=mean([r['regions_lb'] for r in summary['T3'].values()]);c4=mean([r['cells'] for r in clean]);r4lb=mean([r['regions_lb'] for r in clean]);o4=mean([r['ops'] for r in clean]);five=nomass['H8_T5_zoo0_recipe108'];four=nomass['H8_T4_zoo0_recipe108'];c5=five['cells'];r5lb=five['regions_lb']
    trends=dict(T3_mean_cells=c3,T3_mean_exact_regions=r3exact,T3_mean_sampled_regions=r3lb,T3_exact_ratio=c3/r3exact,T3_sampled_ratio_upper=c3/r3lb,T4_mean_cells=c4,T4_mean_regions_lower=r4lb,T4_ratio_upper=c4/r4lb,T5_seed0_cells=c5,T5_seed0_regions_lower=r5lb,T5_ratio_upper=c5/r5lb,cells_mean3_mean4=c4/c3,cells_mean4_seed0T5=c5/c4,matched_seed0_cells_T4_T3=raw['zoo0']['cells']/T3['0']['clean_cells'],matched_seed0_cells_T5_T4=c5/raw['zoo0']['cells'],matched_seed0_nonmass_T5_T4=five['ops_nomass']/four['ops_nomass'],mixed_meanT4total_seed0T5nonmass=five['ops_nomass']/o4,matched_seed0_totalT4_to_nonmassT5=five['ops_nomass']/raw['zoo0']['ops'],T4_seed0_nonmass_fraction=four['ops_nomass']/raw['zoo0']['ops'],assumed13x_five_step_projection=five['ops_nomass']*13**5)
    # A logical counterexample, not an estimate of actual unobserved counts.
    hypothetical=[r3exact,6000,200000];assert hypothetical[1]>=r4lb and hypothetical[2]>=r5lb;ratios=[c3/hypothetical[0],c4/hypothetical[1],c5/hypothetical[2]];assert ratios[0]>ratios[1]>ratios[2]
    out=dict(rows=rows,trends=trends,region_inference_counterexample={'hypothetical_true_regions':hypothetical,'decreasing_cell_ratios':ratios,'meaning':'compatible with saved lower bounds only; not claimed actual counts'},same_unit_scope='Nominal leaderboard fpe666.6667 reproduced; estT counter not certified as complete scalar/hardware cost.',uncertainty_scope='Bonferroni exact binomial intervals conditional on9 fixed models; common-input correlations do not invalidate union bound; no population/noise-distribution claim.',cpu_seconds=time.process_time()-stamp)
    (HERE/'audit_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'all_raw_checks':True,'minimum_simultaneous_ratio':min(r['ratio_over_interval'][0] for r in rows),'trends':trends,'cpu_seconds':out['cpu_seconds']},indent=2))
if __name__=='__main__':main()
