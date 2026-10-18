"""T255 fixed scalar saved audit only. No author, NumPy, model or sampling calls."""
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

P=Path(__file__).resolve().parent
OUT=P/'saved_output'
START=time.process_time()
RESULT={'complete':False,'pass_all':False}
OWNED=False

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save():
    tmp=OUT/'results.tmp'
    with tmp.open('w') as f:
        json.dump(RESULT,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,OUT/'results.json')
def event(kind,**kw):
    with (OUT/'progress.jsonl').open('a') as f:
        f.write(json.dumps(dict(kind=kind,cpu=time.process_time()-START,**kw),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    assert time.process_time()-START<=17

def enc(q):return [str(q.numerator),str(q.denominator)]
def root(q):
    with localcontext() as c:
        c.prec=80
        return (Decimal(q.numerator)/Decimal(q.denominator)).sqrt()
def decfloat(x):return Decimal.from_float(float(x))
def integer(x):return type(x) is int

def main():
    global OWNED
    assert not OUT.exists(),'immutable output exists'
    OUT.mkdir();OWNED=True;event('start')
    protocol=json.loads((P/'saved_protocol.json').read_text());gate=json.loads((P/'saved_gate.json').read_text())
    assert gate['execution_allowed'] is True
    for pins in [protocol['pins'],gate['pins']]:
        for path,h in pins.items():assert sha(path)==h,path
    assert set(gate['reviews'])=={'written_source','checker_source_resource'}
    for role,r in gate['reviews'].items():
        assert r['commit'] and r['publication'] and gate['pins'][r['path']]==r['sha256']==sha(r['path'])
        obj=json.loads(Path(r['path']).read_text());assert obj['complete'] is True and obj['pass_all'] is True
        if role=='checker_source_resource':assert obj['execution_allowed'] is True
    for path in [P/'saved_check.py',P/'saved_protocol.json',P/'manifest.json',P/'SAVED_PROTOCOL.md']:
        assert gate['pins'][str(path)]==sha(path)
    RESULT.update(protocol_sha256=sha(P/'saved_protocol.json'),gate_sha256=sha(P/'saved_gate.json'),limits={k:list(resource.getrlimit(v)) for k,v in [('cpu',resource.RLIMIT_CPU),('as',resource.RLIMIT_AS)]})
    save();assert all(0<x<=20 for x in RESULT['limits']['cpu']) and all(0<x<=2*1024**3 for x in RESULT['limits']['as'])
    m=json.loads((P/'manifest.json').read_text());assert len(m['files'])==20
    for f in m['files']:assert sha(P/f['snapshot'])==f['sha256']
    event('pins_checked')
    rows=[json.loads(l) for l in (P/'snapshot/results/mc_target/blocks.jsonl').read_text().splitlines()]
    assert len(rows)==300 and [r['block'] for r in rows]==list(range(300))
    for b,r in enumerate(rows):
        assert set(r)=={'block','seed','n','correct','correct_float32','sec'}
        assert all(integer(r[k]) for k in ['block','seed','n','correct'])
        assert r['seed']==20261018000+b and r['n']==10**9 and 0<=r['correct']<=r['n']
        assert isinstance(r['sec'],(float,int)) and not isinstance(r['sec'],bool) and math.isfinite(r['sec']) and r['sec']>=0
        if b==0:assert integer(r['correct_float32']) and 0<=r['correct_float32']<=r['n']
        else:assert r['correct_float32'] is None
    event('mc_schema_checked',records=300)
    N=sum(r['n'] for r in rows);K=sum(r['correct'] for r in rows);p=F(K,N);B=len(rows)
    rates=[F(r['correct'],r['n']) for r in rows];mean=sum(rates,F(0))/B
    between=(sum((x-mean)**2 for x in rates)/(B*(B-1)))
    binomial=p*(1-p)/N
    summary=json.loads((P/'snapshot/results/mc_target/summary.json').read_text())
    assert set(summary)=={'blocks','n','correct','acc','se','between_block_sd_of_mean','float32_minus_float64_block0'}
    assert all(integer(summary[k]) for k in ['blocks','n','correct','float32_minus_float64_block0'])
    assert (summary['blocks'],summary['n'],summary['correct'])==(B,N,K)
    assert N==300*10**9 and abs(F(summary['acc'])-p)<=F(1,2**52)
    se=root(binomial);sdb=root(between)
    assert abs(decfloat(summary['se'])-se)<=Decimal(2)**-60
    # Exact rational block rates vs NumPy's rounded rates and reduction: fixed absolute SE tolerance.
    assert abs(decfloat(summary['between_block_sd_of_mean'])-sdb)<=Decimal(2)**-55
    delta=rows[0]['correct_float32']-rows[0]['correct'];assert summary['float32_minus_float64_block0']==delta
    RESULT['mc']={'records':B,'n':N,'correct':K,'accuracy_exact':enc(p),'binomial_variance_exact':enc(binomial),'binomial_se_decimal80':str(se),'between_block_mean_variance_exact':enc(between),'between_block_se_decimal80':str(sdb),'net_float32_correct_delta_block0':delta,'individual_float32_disagreements_unknown':True,'iid_and_runtime_provenance_not_verified':True}
    save();event('mc_arithmetic_checked')
    base=P/'snapshot/results/prefix'
    k5=json.loads((base/'k5.json').read_text());k5m=json.loads((base/'k5_masses.json').read_text());k6=json.loads((base/'k6_masses.json').read_text())
    assert k5['complete'] is True and k5m['complete'] is True and k6['complete'] is False
    for k in ['k','subtrees','subtrees_done','cells','regions']:assert k5[k]==k5m[k]
    for r in [k5,k5m,k6]:
        assert all(integer(r[k]) and r[k]>0 for k in ['subtrees','subtrees_done','cells','regions','workers'])
        assert r['subtrees_done']<=r['subtrees'] and r['regions']<=r['cells']
    assert k5['subtrees_done']==k5['subtrees'] and k6['subtrees_done']<k6['subtrees']
    assert (k6['subtrees'],k6['subtrees_done'])==(3811,245)
    extrap=F(k6['cells']*k6['subtrees'],k6['subtrees_done']);ratio=extrap/k5['cells']
    assert abs(F(k6['cells_est'])-extrap)<=4*F(math.ulp(float(extrap)))
    assert set(k6['N_delta'])=={'0.1','0.01','0.001','0.0001','1e-05','1e-06'}
    counts=[k6['N_delta'][k] for k in ['0.1','0.01','0.001','0.0001','1e-05','1e-06']]
    assert all(integer(x) and 0<x<=k6['regions'] for x in counts) and counts==sorted(counts)
    retained=1+F(k6['leafsum_minus_1']);assert 0<retained<1
    assert k6['sec_wall']>2400 and math.isfinite(k6['sec_cpu'])
    RESULT['prefix']={'k5_cells':k5['cells'],'k5_regions':k5['regions'],'k6_completed_cells':k6['cells'],'k6_completed_histories':k6['regions'],'subtrees_total':k6['subtrees'],'subtrees_done':k6['subtrees_done'],'cells_extrapolation_exact':enc(extrap),'descriptive_k6_extrapolated_cells_over_k5_cells':enc(ratio),'numerical_retained_mass_from_saved_field':enc(retained),'unbiasedness_established':False,'exact_geometry_verified':False,'coverage_counts_recomputed_from_arrays':False,'sec_cpu_is_worker_wall_sum':True}
    save();event('prefix_scalar_checked')
    RESULT['complete']=RESULT['pass_all']=True
    RESULT['author_model_rng_calls']=0
    RESULT['cpu_main']=time.process_time()-START
    event('complete');save()

if __name__=='__main__':
    try:main()
    except BaseException as e:
        RESULT.update(complete=False,pass_all=False,fatal={'type':type(e).__name__,'message':str(e)})
        if OWNED:
            save()
            with (OUT/'progress.jsonl').open('a') as f:
                f.write(json.dumps({'kind':'fatal','type':type(e).__name__,'message':str(e)})+'\n');f.flush();os.fsync(f.fileno())
        raise
