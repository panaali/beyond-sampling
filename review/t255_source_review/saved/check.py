"""Independent integer-moment and scalar audit of T255 saved records only."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal as D,localcontext
import hashlib,json,math,resource,time,traceback
P=Path(__file__).resolve().parent
START=time.process_time()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checkpins(v):
    assert v
    for p,h in v.items():assert sha(p)==h,p
def enc(v):return [str(v.numerator),str(v.denominator)]
def root(v):
    with localcontext() as c:
        c.prec=100
        return (D(v.numerator)/D(v.denominator)).sqrt()
def save(r):
    r['review_cpu']=time.process_time()-START;r['review_maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t=P/'results.tmp';t.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');t.replace(P/'results.json')
def run(r):
    scope=read(P/'scope.json');gate=read(P/'gate.json')
    assert gate['ready'] is True and gate['review_publication'] and gate['execution_publication']
    review=read(gate['source_review']);assert review['complete'] is True and review['pass_all'] is True and review['execution_allowed'] is True
    required={str(P/'check.py'),str(P/'scope.json')}
    assert required<=set(review['pins']) and required|{gate['source_review'],str(P/'completed_inputs.json')}<=set(gate['pins'])
    for v in [scope['pins'],gate['pins'],review['pins']]:checkpins(v)
    completed=read(P/'completed_inputs.json');checkpins(completed['pins']);assert completed['raw_commit'] and completed['publication']
    a=Path(scope['author']);required_author={str(a/n) for n in ['saved_check.py','saved_protocol.json','saved_gate.json','SAVED_PROTOCOL.md','manifest.json','saved_output/results.json','saved_output/progress.jsonl','snapshot/results/mc_target/blocks.jsonl','snapshot/results/mc_target/summary.json','snapshot/results/prefix/k5.json','snapshot/results/prefix/k5_masses.json','snapshot/results/prefix/k6_masses.json']}
    assert required_author<=set(completed['pins'])
    ap=read(a/'saved_protocol.json');ag=read(a/'saved_gate.json');ar=read(a/'saved_output/results.json')
    assert ag['execution_allowed'] is True and ar['complete'] is True and ar['pass_all'] is True
    for v in [ap['pins'],ag['pins']]:checkpins(v)
    assert set(ag['reviews'])=={'written_source','checker_source_resource'}
    for kind,v in ag['reviews'].items():
        assert v['commit'] and v['publication'] and sha(v['path'])==v['sha256']==ag['pins'][v['path']]
        rr=read(v['path']);assert rr['complete'] is True and rr['pass_all'] is True;checkpins(rr['pins'])
        if kind=='checker_source_resource':assert rr['execution_allowed'] is True
    assert ar['protocol_sha256']==sha(a/'saved_protocol.json') and ar['gate_sha256']==sha(a/'saved_gate.json')
    assert ar['limits']==dict(cpu=[20,20],**{'as':[2*1024**3]*2})
    assert resource.getrlimit(resource.RLIMIT_CPU)==(19,20) and resource.getrlimit(resource.RLIMIT_AS)==(2*1024**3,)*2
    assert math.isfinite(ar['cpu_main']) and ar['cpu_main']>=0 and ar['author_model_rng_calls']==0
    manifest=read(a/'manifest.json');assert len(manifest['files'])==20
    for f in manifest['files']:assert sha(a/f['snapshot'])==f['sha256']
    rows=[json.loads(s) for s in (a/'snapshot/results/mc_target/blocks.jsonl').read_text().splitlines()]
    assert len(rows)==300
    total=0;squares=0;seconds=[]
    for i,v in enumerate(rows):
        assert set(v)=={'block','seed','n','correct','correct_float32','sec'}
        assert all(type(v[k])is int for k in ['block','seed','n','correct'])
        assert v['block']==i and v['seed']==20261018000+i and v['n']==10**9 and 0<=v['correct']<=v['n']
        assert type(v['sec']) in (int,float) and math.isfinite(v['sec']) and v['sec']>=0
        if i==0:assert type(v['correct_float32'])is int and 0<=v['correct_float32']<=v['n']
        else:assert v['correct_float32'] is None
        total+=v['correct'];squares+=v['correct']**2;seconds.append(v['sec'])
    blocks=300;n=10**9;N=blocks*n
    accuracy=F(total,N);binvar=F(total*(N-total),N**3)
    # Integer raw moments, rather than centered rational rates.
    between=F(blocks*squares-total**2,blocks**2*(blocks-1)*n**2)
    delta=rows[0]['correct_float32']-rows[0]['correct']
    mr=ar['mc']
    expected=dict(records=blocks,n=N,correct=total,accuracy_exact=enc(accuracy),binomial_variance_exact=enc(binvar),between_block_mean_variance_exact=enc(between),net_float32_correct_delta_block0=delta,individual_float32_disagreements_unknown=True,iid_and_runtime_provenance_not_verified=True)
    assert {k:mr[k] for k in expected}==expected
    roots=[root(binvar),root(between)]
    for name,q,z in zip(['binomial_se_decimal80','between_block_se_decimal80'],[binvar,between],roots,strict=True):
        v=D(mr[name]);assert v.is_finite() and v>0
        with localcontext() as c:
            c.prec=110
            # Two rounded Decimal80 operations (division and sqrt) incur much
            # less than 1e-78 relative error; the independent root uses100 digits.
            assert abs(v-z)<=abs(z)*D(10)**-78
    summary=read(a/'snapshot/results/mc_target/summary.json')
    assert set(summary)=={'blocks','n','correct','acc','se','between_block_sd_of_mean','float32_minus_float64_block0'}
    assert all(type(summary[k])is int for k in ['blocks','n','correct','float32_minus_float64_block0'])
    assert (summary['blocks'],summary['n'],summary['correct'],summary['float32_minus_float64_block0'])==(blocks,N,total,delta)
    assert abs(F(summary['acc'])-accuracy)<=F(1,2**52)
    with localcontext() as c:
        c.prec=110
        assert abs(D.from_float(summary['se'])-roots[0])<=D(2)**-60
        assert abs(D.from_float(summary['between_block_sd_of_mean'])-roots[1])<=D(2)**-55
    r['mc_checked']=dict(**expected,integer_sum_of_squared_hits=str(squares),independent_decimal100_roots=list(map(str,roots)))
    save(r)
    b=a/'snapshot/results/prefix';k5=read(b/'k5.json');k5m=read(b/'k5_masses.json');k6=read(b/'k6_masses.json')
    assert k5['complete'] is True and k5m['complete'] is True and k6['complete'] is False
    assert k5['k']==k5m['k']==5 and k6['k']==6
    for k in ['k','subtrees','subtrees_done','cells','regions']:assert k5[k]==k5m[k]
    for v in [k5,k5m,k6]:
        assert all(type(v[k])is int and v[k]>0 for k in ['subtrees','subtrees_done','cells','regions','workers'])
        assert v['regions']<=v['cells'] and v['subtrees_done']<=v['subtrees']
    assert k5['subtrees_done']==k5['subtrees'] and (k6['subtrees'],k6['subtrees_done'])==(3811,245)
    extrap=F(k6['cells']*3811,245);growth=extrap/k5['cells'];retained=F(k6['leafsum_minus_1'])+1
    expected=dict(k5_cells=k5['cells'],k5_regions=k5['regions'],k6_completed_cells=k6['cells'],k6_completed_histories=k6['regions'],subtrees_total=3811,subtrees_done=245,cells_extrapolation_exact=enc(extrap),descriptive_k6_extrapolated_cells_over_k5_cells=enc(growth),numerical_retained_mass_from_saved_field=enc(retained),unbiasedness_established=False,exact_geometry_verified=False,coverage_counts_recomputed_from_arrays=False,sec_cpu_is_worker_wall_sum=True)
    assert ar['prefix']==expected and abs(F(k6['cells_est'])-extrap)<=4*F(math.ulp(float(extrap)))
    names=['0.1','0.01','0.001','0.0001','1e-05','1e-06'];assert set(k6['N_delta'])==set(names)
    cover=[k6['N_delta'][s] for s in names];assert all(type(x)is int and 0<x<=k6['regions'] for x in cover) and cover==sorted(cover)
    assert 0<retained<1 and k6['sec_wall']>2400 and math.isfinite(k6['sec_cpu'])
    events=[json.loads(s) for s in (a/'saved_output/progress.jsonl').read_text().splitlines()]
    assert [e['kind'] for e in events]==['start','pins_checked','mc_schema_checked','mc_arithmetic_checked','prefix_scalar_checked','complete']
    assert all(set(e)==({'kind','cpu','records'} if e['kind']=='mc_schema_checked' else {'kind','cpu'}) for e in events)
    assert events[2]['records']==300 and all(math.isfinite(e['cpu']) and e['cpu']>=0 for e in events)
    assert all(a['cpu']<=b['cpu'] for a,b in zip(events,events[1:])) and ar['cpu_main']<=events[-1]['cpu']<=20
    assert time.process_time()-START<19
    r.update(complete=True,pass_all=True,records=300,events=6,prefix_checked=expected,author_cpu=ar['cpu_main'],input_pins=completed['pins'],author_calls=0,model_rng_array_calls=0,limitation='Exact saved scalar arithmetic only; capped extrapolation remains completion-selected, masses numerical, MC independence and historical runtime/model identity unproved. No new reference, topology, coverage-array or confidence-interval claim.')
if __name__=='__main__':
    assert not(P/'results.json').exists() and not(P/'results.tmp').exists()
    result=dict(complete=False,pass_all=False);save(result)
    try:run(result)
    except BaseException as e:result.update(complete=False,pass_all=False,error=repr(e),traceback=traceback.format_exc());raise
    finally:save(result)
