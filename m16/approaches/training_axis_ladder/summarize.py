"""Frozen T75-matched ratios and predeclared cost-sensitive comparison."""
import json,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent

def run():
    start=time.process_time();pilot=json.loads((HERE/'pilot.json').read_text());out={'power':29,'noise_seeds':[20,21,22,23],'copies_per_noise_seed':256,'seeds':[],'operation_convention':'Inherited T75 strict primitive units, 72 per M4T3 forward; not a new portable FLOP audit','truth_qualification':'Floating polygon enumeration, independently checked on controls; no formal interval enclosure'}
    full_cpu=0.;hashes=set()
    for seed in range(5):
        arrays=[];panels=[]
        for ns in (20,21,22,23):
            block=json.loads((HERE/f'block_s{seed}_n{ns}.json').read_text());assert block['complete'] and len(block['rows'])==256 and block['power']==29 and block['training_seed']==seed and block['noise_seed']==ns
            assert [r[0] for r in block['rows']]==list(range(256));full_cpu+=block['cpu_seconds']
            for row in block['rows']:
                assert row[-1] not in hashes;hashes.add(row[-1])
            a=np.asarray([r[:11] for r in block['rows']],float);arrays.append(a)
            e=a[:,2]-a[:,1];p=a[:,1];ops=a[:,3]
            panels.append(dict(noise_seed=ns,ratio=float(np.mean(e*e)*np.mean(ops)/(72*np.mean(p*(1-p)))),matched_ratio=float(np.mean(e*e)/np.mean(72*p*(1-p)/ops)),cpu_seconds=block['cpu_seconds']))
        a=np.concatenate(arrays);p=a[:,1];e=a[:,2]-p;ops=a[:,3];pq=p*(1-p);meanops=np.mean(ops);mse=np.mean(e*e)
        ratio=mse*meanops/(72*np.mean(pq));matched=mse/np.mean(72*pq/ops)
        idx=np.random.default_rng(0).integers(0,len(a),size=(500,len(a)));bs=np.mean(e[idx]**2,1)*meanops/(72*np.mean(pq[idx],1));bsm=np.mean(e[idx]**2,1)/np.mean(72*pq[idx]/ops[idx],1)
        clean=next(r for r in pilot['rows'] if r['power']==29 and r['training_seed']==seed and r['copy']==-1)
        ci=np.quantile(bs,[.025,.975]);cim=np.quantile(bsm,[.025,.975]);single=e*e/(72*pq/ops)
        row=dict(training_seed=seed,clean_truth=clean['truth']['value'],clean_error=clean['error'],clean_ratio=clean['ratio'],noisy_mean_truth=float(np.mean(p)),rmse=float(np.sqrt(mse)),bias=float(np.mean(e)),max_abs_error=float(np.max(abs(e))),mean_candidate_ops=float(meanops),mean_reference_ops=float(np.mean(a[:,4])),candidate_cells_mean=float(np.mean(a[:,8])),reference_correct_leaves_mean=float(np.mean(a[:,5])),resolved_fraction=float(np.sum(a[:,7])/np.sum(a[:,6])),mse_ratio=float(ratio),ratio_ci=ci.tolist(),model_matched_ratio=float(matched),model_matched_ci=cim.tolist(),integer_budget_matched_ratio=float(mse/np.mean(pq/np.floor(ops/72))),single_copy_ratio_max=float(np.max(single)),single_copy_ratios_above_one=int(np.sum(single>1)),panels=panels,passes_frozen_criterion=bool(clean['ratio']<1 and ratio<1 and ci[1]<1),dd_clean_gate_cells=clean['independent']['counts'][2])
        out['seeds'].append(row)
    out.update(all_seed_criteria_pass=all(r['passes_frozen_criterion'] for r in out['seeds']),validation_block_cpu_seconds=full_cpu,pilot_cpu_seconds=pilot['cpu_seconds'],unique_noisy_models=len(hashes),bootstrap_replicates_per_seed=500,bootstrap_index_draws=5*500*1024,summary_cpu_seconds=time.process_time()-start)
    (HERE/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    for r in out['seeds']:print(r['training_seed'],r['mse_ratio'],r['ratio_ci'],r['model_matched_ratio'],'maxcopy',r['single_copy_ratio_max'],flush=True)
    print('all criteria',out['all_seed_criteria_pass'],'CPU',full_cpu,flush=True)
if __name__=='__main__':run()
