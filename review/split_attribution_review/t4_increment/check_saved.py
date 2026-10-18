"""Fixed incremental saved records only; no author imports or numerical replay."""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json
import math
import time
import traceback

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def q(x):
    assert type(x) in (int,float) and math.isfinite(x)
    return F(x)
def enc(x):return {'numerator':str(x.numerator),'denominator':str(x.denominator),'float':float(x)}

def main():
    start=time.process_time()
    outputs=[ROOT/n for n in ('results.json','progress.jsonl','failure.json')]
    assert not any(p.exists() for p in outputs)
    gate=read(ROOT/'execution_gate.json')
    assert gate['status']=='completed_source_review' and gate['review_publication']
    for name,digest in gate['pins'].items():
        path=Path(name);path=path if path.is_absolute() else ROOT/path
        assert sha(path)==digest,name
    with outputs[1].open('x') as log:
        def event(**kw):log.write(json.dumps(kw)+'\n');log.flush()
        event(phase='started',gate_sha256=sha(ROOT/'execution_gate.json'))
        try:
            m=read(ROOT/'manifest.json');assert len(m['records'])==22
            files={}
            for v in m['records']:
                path=ROOT/'frozen'/v['original']
                assert sha(path)==v['sha256'] and path.stat().st_size==v['bytes']
                assert v['original'] not in files;files[v['original']]=path
            prior=read(Path(gate['prior_T3_results']))
            assert prior['status']=='pass' and prior['rows_checked']==51
            t3={tuple(v['key']):v for v in prior['rows']};assert len(t3)==51
            base='m16/approaches/split_attribution/'
            raw=sorted(x for x in files if x.startswith(base+'prune/'))
            assert len(raw)==15 and raw==sorted(m['new_T4_paths'])
            truth=read(files['m16/approaches/estT_fix/summary.json'])
            keys=set();rows=[];groups={}
            for name in raw:
                a=read(files[name]);s=a['seed'];test=a['test'];tau=a['tau'];key=(s,test,tau)
                assert a['H']==8 and a['T']==4 and key not in keys;keys.add(key)
                for field in ('ops','flops','retained_cells','resolved','pruned','wrong_centroid'):
                    assert type(a[field]) is int and a[field]>=0
                assert a['flops']==2*a['ops'] and a['wrong_centroid']<=a['pruned']
                assert sum(x[0] for x in a['by_step'].values())==a['pruned']
                for step,x in a['by_step'].items():
                    assert step in ('1','2','3','4') and len(x)==2 and type(x[0]) is int and x[0]>=0
                    q(x[1])
                p=q(truth[f'zoo{s}']['exact']);assert 0<p<1
                acc,b,cost=q(a['acc']),q(a['bound']),q(a['flops'])
                assert 0<=acc<=1 and b>=0 and cost>0 and q(a['sec'])>=0
                draws=cost/682;var=p*(1-p)/draws;ra=(acc-p)**2/var;rb=b*b/var
                # Historical source fpe_flops(8,4)=6000*640/5760.
                old_residual=q(a['N'])-cost/F(2000,3)
                assert abs(old_residual)<=2*q(math.ulp(a['N']))
                pad=q(1e-15)*a['pruned'] if test=='mass' else F(0)
                residual=b-sum((q(x[1]) for x in a['by_step'].values()),F(0))-pad
                ref=read(files[base+f'ref/H8_T4_s{s}.json'])
                old=t3[key]
                entry=dict(name=name,sha256=sha(files[name]),key=list(key),error=enc(acc-p),bound=enc(b),
                    continuous_draws=enc(draws),variance=enc(var),ratio_actual=enc(ra),ratio_bound=enc(rb),
                    historical_N_residual=enc(old_residual),padded_step_residual=enc(residual),
                    cells=a['retained_cells'],unpruned_cells=ref['cells'],T3_cells_same_seed=old['retained_cells'],
                    T3_record_sha256=old['sha256'],flops=a['flops'],author_wall_seconds=a['sec'])
                rows.append(entry);groups.setdefault((test,tau),[]).append((s,ra,rb,a['retained_cells'],old['retained_cells']))
                event(phase='row_checked',name=name)
            taus=(1e-4,1e-5,1e-6,1e-7,1e-8)
            expected={(0,t,v) for t in ('cap','mass') for v in taus}
            expected.update((1,'cap',v) for v in taus[:3]);expected.update((1,'mass',v) for v in taus[:2])
            assert keys==expected
            aggregate=[]
            for (test,tau),values in sorted(groups.items()):
                seeds=sorted(x[0] for x in values);n=len(values)
                aggregate.append(dict(test=test,tau=tau,seeds=seeds,missing_seeds=sorted(set(range(5))-set(seeds)),
                    mean_cells_available=enc(F(sum(x[3] for x in values),n)),
                    max_ratio_actual_available=enc(max(x[1] for x in values)),
                    max_ratio_bound_available=enc(max(x[2] for x in values)),
                    T4_over_T3_cells_matched_seeds=enc(F(sum(x[3] for x in values),sum(x[4] for x in values)))))
            assert len(aggregate)==10 and not m['T5_prune_paths'] and not m['rendered_prune_report_present']
            result=dict(status='pass',manifest_sha256=sha(ROOT/'manifest.json'),checker_sha256=sha(Path(__file__)),
                gate_sha256=sha(ROOT/'execution_gate.json'),rows=rows,groups=aggregate,rows_checked=15,groups_checked=10,
                prior_T3_rows_reused=15,prior_T3_results_sha256=sha(Path(gate['prior_T3_results'])),
                T5_archive_loaded=False,author_calls=0,review_cpu=time.process_time()-start,
                author_wall_seconds=sum(x['author_wall_seconds'] for x in rows),
                scope='Partial seeds/thresholds only; exact stored-float point ratios, continuous budget, reported bound only; no CI/cap/totalcost/scaling certificate.')
            with outputs[0].open('x') as f:json.dump(result,f,indent=2);f.write('\n')
            event(phase='completed',result_sha256=sha(outputs[0]))
        except BaseException:
            with outputs[2].open('x') as f:json.dump(dict(traceback=traceback.format_exc(),review_cpu=time.process_time()-start),f,indent=2)
            event(phase='failed',failure_sha256=sha(outputs[2]));raise

if __name__=='__main__':main()
