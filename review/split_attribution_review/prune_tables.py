"""Saved-only independent exact-budget contraction of all pruning table rows."""
import hashlib,json,math,time,traceback
from fractions import Fraction as F
from pathlib import Path
P=Path(__file__).resolve().parent;D=P/'frozen'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def log(x):
    with (P/'prune_tables_progress.jsonl').open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
def run():
    gate=load(P/'prune_tables_gate.json')
    for rel,h in gate['pins'].items():assert sha(P/rel)==h,rel
    assert not (P/'prune_tables_results.json').exists() and not (P/'prune_tables_progress.jsonl').exists()
    log({'state':'started'});start=time.process_time()
    try:
        saved=load(P/'saved_recovery_results.json');assert len(saved['pruning'])==52
        truth3=load(D/'ref/exact_H8_T3.json');truth4=load(D/'dependencies/m16/approaches/estT_fix/summary.json')
        report=(D/'prune_report.md').read_text();lines=[];groups={}
        for entry in saved['pruning']:
            r=entry['record']
            if r['tau']==0:continue
            T,s=r['T'],r['seed'];p=F(truth3[str(s)] if T==3 else truth4[f'zoo{s}']['exact']);N=F(entry['continuous_budget_exact'])
            e=F(r['acc'])-p;v=p*(1-p)/N;ra=e*e/v;rb=F(r['bound'])**2/v;sd=math.sqrt(float(v));ref=load(D/'ref'/f'H8_T{T}_s{s}.json');bo=r.get('bound_oracle_uncertain')
            line=f"| {T} | {r['test']} | {r['tau']:g} | {s} | {r['retained_cells']} ({ref['cells']}) | {r['flops']:.3g} ({2*ref['ops']:.3g}) | {r['pruned']} | {float(e):+.2e} | {r['bound']:.2e} | {'' if bo is None else f'{bo:.2e}'} | {sd:.2e} | {float(ra):.3g} | {float(rb):.3g} |"
            assert line in report,line
            result={'file':entry['file'],'reference':float(p),'N_exact':str(N),'error_exact':str(e),'variance_exact':str(v),'ratio_act_exact':str(ra),'ratio_B_exact':str(rb),'ratio_act':float(ra),'ratio_B':float(rb),'retained_cells':r['retained_cells'],'line':line}
            lines.append(result);groups.setdefault((r['test'],r['tau'],T),[]).append(result)
        assert len(lines)==51
        pooled=[]
        for method in ['cap','mass']:
            for tau in sorted({key[1] for key in groups if key[0]==method},reverse=True):
                cells={T:math.fsum(x['retained_cells'] for x in groups[(method,tau,T)])/len(groups[(method,tau,T)]) for T in [3,4,5] if (method,tau,T) in groups}
                bs={T:max(x['ratio_B'] for x in groups[(method,tau,T)]) for T in cells};actual={T:max(x['ratio_act'] for x in groups[(method,tau,T)]) for T in cells}
                g34=cells[4]/cells[3] if 3 in cells and 4 in cells else float('nan');g45=cells[5]/cells[4] if 4 in cells and 5 in cells else float('nan')
                line=f"| {method} | {tau:g} | {cells.get(3,float('nan')):.0f} | {cells.get(4,float('nan')):.0f} | {cells.get(5,float('nan')):.0f} | {g34:.1f} | {g45:.1f} | "+', '.join(f'T{T}:{v:.3g}' for T,v in bs.items())+' | '+', '.join(f'T{T}:{v:.3g}' for T,v in actual.items())+' |'
                assert line in report,line
                pooled.append({'method':method,'tau':tau,'cells':cells,'max_ratio_B':bs,'max_ratio_act':actual,'line':line})
        assert len(pooled)==10
        result={'gate_sha256':sha(P/'prune_tables_gate.json'),'rows':lines,'pooled':pooled,'cpu_seconds':time.process_time()-start,'scope':'saved numerical reference points; continuous historical T196 allowance; no full-cost or exact-reference certification'}
        with (P/'prune_tables_results.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
        log({'state':'completed','result_sha256':sha(P/'prune_tables_results.json')})
    except BaseException:log({'state':'failed','traceback':traceback.format_exc()});raise
if __name__=='__main__':run()
