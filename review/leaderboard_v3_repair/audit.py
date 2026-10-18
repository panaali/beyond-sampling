"""T204 saved-only audit; imports no author modules and evaluates no model."""
import ast
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import resource
import time
HERE=Path(__file__).resolve().parent

def read(path): return json.loads((HERE/path).read_text())
def classes(path):
    tree=ast.parse((HERE/path).read_text())
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CLASS' for t in n.targets))
def status(row):
    if 'det_tested' not in row:
        return 'legacy_recorded_pass' if row.get('det_ok') is True else 'legacy_unknown'
    if row['det_tested'] is False: return 'not_tested'
    if row['det_tested'] is True and type(row.get('det_ok')) is bool:
        return 'tested_pass' if row['det_ok'] else 'tested_fail'
    raise ValueError('invalid explicit determinism metadata')
def ratio(errors,truths,costs):
    assert len(errors)==len(truths)==len(costs) and len(errors)>0
    assert all(c>0 for c in costs)
    return sum(e*e for e in errors)/sum(p*(1-p)/c for p,c in zip(truths,costs,strict=True))
def controls():
    rows=[({'det_ok':False},'legacy_unknown'),({'det_ok':True},'legacy_recorded_pass'),({'det_tested':False,'det_ok':False},'not_tested'),({'det_tested':True,'det_ok':False},'tested_fail'),({'det_tested':True,'det_ok':True},'tested_pass')]
    out=[]
    for r,want in rows:
        got=status(r);assert got==want;out.append(dict(input=r,expected=want,actual=got))
    p=[F(1,2),F(1,4)];e=[F(1,8)]*2
    a,b,u=ratio(e,p,[F(1),F(3)]),ratio(e,p,[F(3),F(1)]),ratio(e,p,[F(2)]*2)
    assert (a,b,u)==(F(1,10),F(3,26),F(1,7))
    return dict(determinism=out,cost_covariance=dict(own_a=str(a),own_b=str(b),uniform=str(u)))
def load_raw(path):
    rows={};duplicates=[]
    for line in (HERE/path).read_text().splitlines():
        d=json.loads(line)
        if (d['v'],d['H'],d['T'])==(3,16,10):
            key=(d['name'],d['model'],d['budget'],d['rep'])
            if key in rows:
                digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
                duplicates.append(dict(key=list(key),identical=rows[key]==d,previous_sha=digest(rows[key]),replacement_sha=digest(d)))
            rows[key]=d
    return rows,duplicates

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(19,20));start=time.process_time()
    target=HERE/'results.json'
    assert not any((HERE/n).exists() for n in ('results.json','all_budgets.json','ALL_BUDGETS.md')),'immutable first run'
    out=dict(status='started',source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def save():target.write_text(json.dumps(out,indent=2)+'\n')
    save()
    try:
        gate=read('gate.json');assert gate['approved'] is True
        gp=Path(gate['artifact']);assert hashlib.sha256(gp.read_bytes()).hexdigest()==gate['sha256']
        protocol=read('protocol.json')
        assert out['source_sha']==protocol['audit_sha256']
        assert hashlib.sha256((HERE/'manifest.json').read_bytes()).hexdigest()==protocol['manifest_sha256']
        manifest=read('manifest.json')
        for r in manifest['files']:assert hashlib.sha256((HERE/r['snapshot']).read_bytes()).hexdigest()==r['sha256']
        out['controls']=controls();save()
        current=classes('frozen/current/m16/leaderboard.py');old=classes('frozen/pre_class/m16/leaderboard.py')
        raw,duplicates=load_raw('frozen/current/results/leaderboard_v3_raw.jsonl');prior,prior_duplicates=load_raw('frozen/pre_class/results/leaderboard_v3_raw.jsonl')
        out['raw_duplicate_inventory']=dict(policy='last record per key, matching pinned load_raw source; every duplicate retained here',current=duplicates,pre_class=prior_duplicates)
        inventory=[]
        for name in sorted({k[0] for k in raw}):
            items=[(k,v) for k,v in raw.items() if k[0]==name]
            inventory.append(dict(name=name,current_class=current.get(name,['unreviewed'])[0],pre_class=old.get(name,['unreviewed'])[0],rows=len(items),states=dict(Counter(status(v) for _,v in items)),identical_pre_class_records=sum(prior.get(k)==v for k,v in items)))
        out['determinism_inventory']=inventory;save()
        rows=read('frozen/current/results/leaderboard_v3_H16_T10.json')
        assert len(rows)==498==protocol['records']['archived_H16T10_report']
        keys=[(r['name'],r['group'],r['budget']) for r in rows];assert len(keys)==len(set(keys))
        exported=[]
        for r in rows:
            x=dict(r);x['dri_class_at_source_snapshot']=current.get(r['name'],['unreviewed'])[0];x['determinism_evidence']='See method-wide inventory; legacy False remains unknown';exported.append(x)
        assert all(all(x[k]==v for k,v in r.items()) for r,x in zip(rows,exported,strict=True))
        (HERE/'all_budgets.json').write_text(json.dumps(exported,indent=2)+'\n')
        md=['# Frozen H16T10 all-budget evidence','', 'Every archived row is retained. Ratios, intervals, classes, tuned flags, setup and timing fields are copied, not recomputed. Current DRI class is separate from archived reporting class. Legacy False does not establish a failed test. These archived bootstrap intervals do not fully propagate shared truth uncertainty; this is not a new rung pass.','', '| Method | Group | Budget | Archived class | DRI class | Ratio | Archived 95% CI | Tuned |','|---|---|---:|---|---|---:|---|---|']
        for r in sorted(exported,key=lambda x:(x['name'],x['group'],x['budget'])):
            md.append(f"| {r['name']} | {r['group']} | {r['budget']} | {r['cls']} | {r['dri_class_at_source_snapshot']} | {r['ratio']:.12g} | [{r['ci'][0]:.12g}, {r['ci'][1]:.12g}] | {r['tuned']} |")
        (HERE/'ALL_BUDGETS.md').write_text('\n'.join(md)+'\n')
        out['report']=dict(rows=len(rows),unique_keys=len(set(keys)),all_original_fields_preserved=all(all(x[k]==v for k,v in r.items()) for r,x in zip(rows,exported)),negative_ratios=sum(r['ratio']<0 for r in rows),degenerate_intervals=sum(r['ci'][0]==r['ci'][1] for r in rows))
        p=read('frozen/current/m16/approaches/ladder_training_axis/rung_k24.json')['seeds']['4']['panel']
        ee=list(map(F,p['t75_errs']));pp=list(map(F,p['truths']));cost=F(p['t75']['ops'])/72
        assert len(ee)==len(pp)==1024==protocol['records']['E44_k24_seed4']
        old_ratio=ratio(ee,pp,[cost]*len(ee));assert abs(float(old_ratio)-p['t75']['ratio'])<1e-12
        d=read('frozen/current/m16/approaches/training_axis_review/absolute_k24_s4.json');rr=d['rows']
        assert len(rr)==256==protocol['records']['T114_absolute_k24_seed4']
        errors=[F(r['t75']['value'])-F(r['truth']) for r in rr];truths=[F(r['truth']) for r in rr];costs=[F(r['t75']['own_ops'],r['t75']['sample_ops']-3) for r in rr]
        own=ratio(errors,truths,costs);uniform=ratio(errors,truths,[sum(costs)/len(costs)]*len(costs))
        summary=next(r for r in read('frozen/current/m16/approaches/training_axis_review/summary.json')['cells'] if r['file']=='absolute_k24_s4.json')
        assert abs(float(own)-summary['t75']['protocol_aligned_ratio'])<1e-12
        assert abs(float(uniform)-summary['t75']['protocol_uniform_mean_cost_ratio'])<1e-12
        out['e44_t114']=dict(e44_n=len(ee),e44_uniform_ratio=str(old_ratio),e44_ratio=float(old_ratio),e44_saved_ci=p['t75']['ci'],e44_missing_per_copy_costs=True,t114_n=len(rr),t114_own_ratio=str(own),t114_ratio=float(own),t114_uniform_ratio=float(uniform),t114_saved_ci=summary['t75']['protocol_aligned_bootstrap_ci'],t114_noise_seeds=d['noise_seeds'],same_panel=False,transfer_factor_validated=False)
        out['status']='complete'
    except BaseException as exc:
        out['status']='failed';out['exception']=repr(exc);raise
    finally:
        out['cpu_seconds']=time.process_time()-start;save()
if __name__=='__main__':main()
