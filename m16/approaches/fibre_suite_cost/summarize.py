"""Equal-weight fixed-panel ratios with per-model matched expression budgets."""
import argparse,json,math,statistics
from pathlib import Path
from collections import defaultdict
HERE=Path(__file__).parent

def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups['clean' if r['name'].startswith('seed') else r['name'].rsplit('_',1)[0]].append(r)
    out=dict(models=len(rows),all_exact_parity=all(all(r['parity'].values()) for r in rows),max_individual_ratio=max(r['squared_bias_ratio'] for r in rows),worst=max(rows,key=lambda r:r['squared_bias_ratio'])['name'],individual_ratio_ge_one=[r['name'] for r in rows if r['squared_bias_ratio']>=1],budget_multiplier_range=[min(r['equivalent_budget_multiplier'] for r in rows),max(r['equivalent_budget_multiplier'] for r in rows)],mean_budget_multiplier=statistics.mean(r['equivalent_budget_multiplier'] for r in rows),mean_expression_cost=statistics.mean(r['native']['expression_total'] for r in rows),mean_original_counter=statistics.mean(r['original_counter'] for r in rows),groups={})
    for name,rs in sorted(groups.items()):
        mse=math.fsum(r['bias']**2 for r in rs)/len(rs);var=math.fsum(r['sampling_variance'] for r in rs)/len(rs);oldvar=math.fsum(r['truth']*(1-r['truth'])/r['original_sample_equivalents'] for r in rs)/len(rs)
        out['groups'][name]=dict(models=len(rs),mean_squared_bias=mse,mean_matched_expression_sampling_variance=var,ratio_of_means=mse/var,old_mean_sampling_variance=oldvar,old_ratio_of_means=mse/oldvar,max_individual_ratio=max(r['squared_bias_ratio'] for r in rs))
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args();allrows={};chunks=[]
    for path in sorted(HERE.glob('h[456]_*.json')):
        r=json.loads(path.read_text());assert r['complete'],f'Incomplete {path.name}';chunks.append(dict(file=path.name,H=r['H'],models=len(r['rows']),cpu=r['cpu_including_imports']))
        for row in r['rows']:
            key=(r['H'],row['index']);assert key not in allrows and 'failure' not in row;allrows[key]=row
    out=dict(by_H={},models=len(allrows),complete=len(allrows)==471,scope='Ratio of equal-weight empirical mean squared bias to equal-weight per-model matched-budget sampling variance; not mean individual ratios or one fixed common budget.',new_truth_or_noise=False,full_truth_interval_certificate=False,complete_implementation_cost=False,chunks=chunks)
    for h in (4,5,6):
        rows=[allrows[(h,i)] for i in range(157) if (h,i) in allrows]
        if rows:out['by_H'][str(h)]=summarize(rows)
    if not args.partial:assert set(allrows)=={(h,i) for h in (4,5,6) for i in range(157)}
    dest=HERE/('summary_partial.json' if args.partial else 'summary.json');dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({h:{k:v for k,v in r.items() if k!='groups'} for h,r in out['by_H'].items()},indent=2))
if __name__=='__main__':main()
