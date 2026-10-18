"""Saved evidence only; no estimator imports, model calls or new random draws."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal
import ast,copy,json,hashlib,time,resource,math
import scipy
from scipy.stats import beta
HERE=Path(__file__).resolve().parent;S=HERE/'snapshot';E=S/'m16/approaches/estT_fix'
def read(p):return json.loads(Path(p).read_text(),parse_float=Decimal)
def q(x):return x if isinstance(x,F) else F(x)
def show(x):return dict(fraction=str(x),float=float(x))
def risk(e,p,draws):return (e-p)**2*draws/(p*(1-p))
def interval_risk(e,lo,hi,draws):
    assert 0<lo<=hi<1 and 0<e<1
    ends=[risk(e,lo,draws),risk(e,hi,draws)]
    return [F(0) if lo<=e<=hi else min(ends),max(ends)]
def funcs(p):return {n.name:n for n in ast.parse(Path(p).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def dump(n):return ast.dump(n,include_attributes=False)
def main():
    start=time.process_time();out=HERE/'results.json';progress=HERE/'progress.jsonl';assert not out.exists() and not progress.exists()
    P=read(HERE/'protocol.json');assert P['source_gate']
    assert hashlib.sha256((HERE/'manifest.json').read_bytes()).hexdigest()==P['manifest_sha256']
    manifest=read(HERE/'manifest.json')
    for r in manifest['files']:assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
    assert hashlib.sha256((HERE/'audit.py').read_bytes()).hexdigest()==P['audit_sha256']
    result=dict(complete=False,rows=[],source_manifest_sha256=hashlib.sha256((HERE/'manifest.json').read_bytes()).hexdigest(),scipy_version=scipy.__version__,model_mass_MC_calls=0)
    def log(v):
        with progress.open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n')
    def save():
        result.update(cpu=time.process_time()-start,process_cpu=time.process_time(),maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        p=out.with_suffix('.tmp');p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');p.replace(out)
    try:
        # Source parity removes ONLY the dimension-five dispatch, unreachable at T4.
        a=funcs(E/'estT.py');b=funcs(S/'evidence/source_latest.py');same={k:dump(a[k])==dump(v) for k,v in b.items()}
        orth=copy.deepcopy(a['orthant']);remove=[i for i,n in enumerate(orth.body) if isinstance(n,ast.If) and ast.unparse(n.test)=='d == 5']
        assert len(remove)==1 and ast.unparse(orth.body[remove[0]])=='if d == 5:\n    return _orth5(Rays, ops)'
        orth.body.pop(remove[0]);assert dump(orth)==dump(b['orthant'])
        assert all(v for k,v in same.items() if k!='orthant')
        old=ast.parse((S/'evidence/source_latest.py').read_text());new=ast.parse((E/'estT.py').read_text())
        def assignments(tree):return [dump(n) for n in tree.body if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign))]
        oa,na=assignments(old),assignments(new);assert na[:len(oa)]==oa
        result['four_dimensional_source_parity']=dict(functions=same,orthant_after_only_5d_dispatch_removal=True,original_top_level_assignments_prefix_equal=True,extra_current_assignments=len(na)-len(oa),caveat='No historical runtime/source hash or native dispatch binding follows from source parity; extra5D cold setup not priced.')
        # Exact scalar controls of interval extrema, including an interior zero.
        assert interval_risk(F(1,2),F(1,4),F(3,4),F(1))==[F(0),F(1,3)]
        assert interval_risk(F(1,4),F(1,2),F(3,4),F(1))==[F(1,4),F(4,3)]
        assert interval_risk(F(3,4),F(1,4),F(1,2),F(1))==[F(1,4),F(4,3)]
        summary=read(E/'summary.json');mc=read(E/'mc_H8_T4.json');N=10_000_000;alpha=F(1,20);sample=F(2000,3)
        names=sorted([f'zoo{i}' for i in range(5)]+[f'zoo0_noisy{i}' for i in range(4)])
        assert sorted(mc)==names and sorted(k for k in summary if k!='oversplit_by_step')==names
        totals={str(t):[0,0,0] for t in (1,2,3)}
        for name in names:
            log(dict(stage='attempt',name=name));r=read(E/'rescore'/(name+'.json'));s=summary[name];e=q(r['recipe108']['acc']);p=q(r['exact']['acc']);cost=int(r['recipe108']['ops']);draws=F(cost)/sample
            assert q(s['exact'])==p and q(s['est'])==e and int(s['ops'])==cost and q(s['mc1e7'])==q(mc[name])
            ratio=risk(e,p,draws);assert abs(float(ratio)-float(s['ratio']))<=1e-12*(1+float(ratio))
            assert abs(float(100*(e-p))-float(s['err_pp']))<=1e-11
            assert r['recipe108']['cells']==s['cells'] and r['splits']==s['splits']
            hits=round(q(mc[name])*N);assert abs(q(mc[name])-F(hits,N))<F(1,10**15)
            intervals={}
            for label,aerr in [('pointwise95',alpha),('nine_case_bonferroni95',alpha/9)]:
                lo=float(beta.ppf(float(aerr/2),hits,N-hits+1));hi=float(beta.ppf(float(1-aerr/2),hits+1,N-hits));assert 0<lo<hi<1
                residual=[float(beta.cdf(lo,hits,N-hits+1))-float(aerr/2),float(beta.cdf(hi,hits+1,N-hits))-float(1-aerr/2)]
                assert max(map(abs,residual))<1e-7
                intervals[label]=dict(probability=[lo,hi],tail_residuals=residual,legacy_ratio=[show(v) for v in interval_risk(e,F(lo),F(hi),draws)],floor_draw_ratio=[show(v) for v in interval_risk(e,F(lo),F(hi),F(draws.numerator//draws.denominator))])
            B=p*(1-p)/(e-p)**2 if e!=p else None
            row=dict(name=name,estimate=show(e),same_source_reference=show(p),legacy_cost=cost,legacy_sample_cost=show(sample),fractional_draws=show(draws),nominal_ratio=show(ratio),floor_ratio=show(risk(e,p,F(draws.numerator//draws.denominator))),
                source_documented_MC_N=N,reconstructed_MC_hits=hits,MC_raw_hit_ledger_saved=False,MC_intervals=intervals,
                conditional_break_even_draws=show(B) if B is not None else None,
                conditional_break_even_cost_legacy=show(B*sample) if B is not None else None,
                hypothetical_break_even_cost_sample512=show(B*512) if B is not None else None,
                sample512_applicability='Hypothetical headroom under separately reviewed explicit-source sampler convention only; existing legacy/native estimator totals are not converted to these units.',
                historical_binding=dict(model_namespace='H8T4 clean seed'+name[3] if 'noisy' not in name else 'H8T4seed0 relative RMS sigma.005,noise seed7,index'+name[-1],saved_model_weight_hash=False,saved_source_hash=False,source_documented_namespace_only=True),
                diagnostics_cost='recipe wallclock includes uncharged DIAG minority clipping/64 Dirichlet points/forwards; legacy Ops exclude diagnostics, not a clean estimator-only runtime')
            for t,v in r['splits'].items():
                for k,key in enumerate(('n','class_changes','err_changes')):totals[t][k]+=int(v[key])
            result['rows'].append(row);log(dict(stage='result',row=row));save()
        for t,(n,c,e) in totals.items():
            z=summary['oversplit_by_step'][t];assert [z[k] for k in ('splits','class_change','err_change')]==[n,c,e]
            assert abs(float(z['oversplit_class'])-float(F(n-c,n)))<1e-14
        result['diagnostic_totals']=totals
        # Reuse independent positive references without replay or claiming full FP certification.
        zero=read(S/'evidence/positive_seed0.json');one=read(S/'evidence/positive_seed1.json');latest=read(S/'evidence/source_latest_summary.json');cost=read(S/'evidence/cost_summary.json')
        refs=[('zoo0',q(zero['positive_accuracy']),q(zero['conditional_integration_bound'])),('zoo1',q(one['exact']['accuracy_positive']),q(one['exact']['conditional_bound']))]
        result['conditional_positive_reference_comparisons']=[]
        for name,p,eps in refs:
            row=next(r for r in result['rows'] if r['name']==name);e=F(row['estimate']['fraction']);draws=F(row['fractional_draws']['fraction']);old=F(row['same_source_reference']['fraction'])
            result['conditional_positive_reference_comparisons'].append(dict(name=name,positive_reference=show(p),conditional_bound=show(eps),published_reference_minus_positive=show(old-p),legacy_ratio_range=[show(v) for v in interval_risk(e,p-eps,p+eps,draws)],scope='Assumes declared same checkpoint namespace and prior captured geometry; conditional integration only, excludes floating/geometry/historical model binding.'))
        assert q(latest['source_accuracy'])==q(summary['zoo0']['exact'])
        assert latest['legacy_ops_once']==1064714168 and cost['revised_partial_hybrid']==3591731780
        result['cost_evidence']=dict(corpus='seed0 exact35098 error simplices only; NOT recipe/case-family cost',legacy_mass_counter=latest['legacy_ops_once'],partial_hybrid=cost['revised_partial_hybrid'],complete_native_cost=cost['complete_native_cost'],cost_multiplier_transfer_allowed=False,scope=cost['scope'])
        prior=read(S/'evidence/t5_evidence.json')
        assert q(read(E/'run5/zoo0_recipe108.json')['acc'])==F(prior['one_model']['estimate']['fraction'])
        assert q(read(E/'run5/zoo0_exact.json')['acc'])==F(prior['one_model']['numerical_reference']['fraction'])
        result['reused_T5_review']=dict(review='199420a1',nominal_ratio=float(prior['one_model']['nominal_ratio']['float']),pointwise_MC_ratio_interval=list(map(float,prior['mc']['ratio_interval'])),scope=prior['scope'],no_repeated_audit_or_new_truth=True,actual_corpus_phase='BlockedT11584 DRI decision remains separate')
        result['coverage']=dict(clean_seeds=list(range(5)),noisy_base_seeds=[0],noisy_copies=4,not_all_five_seeds_have_noisy_copies=True,MC_stream='Source reinitializes default_rng(2026) for each model; cross-model errors correlated. Bonferroni does not assume independence.',full_rung_confirmed=False)
        result['complete']=True;log(dict(stage='complete',cases=9));save()
    except BaseException as e:result['failure']=repr(e);log(dict(stage='failure',error=repr(e)));save();raise
    print(json.dumps(dict(complete=True,cases=len(result['rows']),cpu=result['cpu'])))
if __name__=='__main__':main()
