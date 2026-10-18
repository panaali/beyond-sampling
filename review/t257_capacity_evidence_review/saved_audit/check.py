"""Independent saved arithmetic; no producer imports, model or binary reads."""
from pathlib import Path
from fractions import Fraction as Q
import hashlib,json,math,os,re,resource,time,traceback

P=Path(__file__).resolve().parent
B=Path('/workspace/work/feynman_fundamentals/m16/approaches/feynman_fundamentals/capacity_retained_evidence')
OUT=P/'output'; START=time.process_time(); OWN=False
R={'complete':False,'pass_all':False}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def pair(x): return [str(x.numerator),str(x.denominator)]
def unpair(x): return Q(int(x[0]),int(x[1]))
def save():
    with (OUT/'partial.tmp').open('w') as f:
        json.dump(R,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(OUT/'partial.tmp',OUT/'results.json')
def event(kind,**kw):
    with (OUT/'progress.jsonl').open('a') as f:
        f.write(json.dumps(dict(kind=kind,cpu=time.process_time()-START,**kw))+'\n');f.flush();os.fsync(f.fileno())
    assert time.process_time()-START<16
def common_sum(values):
    ratios=[v.as_integer_ratio() for v in values]
    den=max(b for a,b in ratios)
    assert all(den%b==0 for a,b in ratios)
    return Q(sum(a*(den//b) for a,b in ratios),den)
def main():
    global OWN
    assert not OUT.exists();OUT.mkdir();OWN=True;event('start')
    protocol=read(P/'protocol.json');gate=read(P/'gate.json')
    assert gate['execution_allowed'] is True
    for pins in [protocol['pins'],gate['pins']]:
        for path,digest in pins.items(): assert sha(path)==digest,path
    direct=[P/'check.py',P/'protocol.json',P/'PROTOCOL.md']
    assert all(gate['pins'][str(p)]==sha(p) for p in direct)
    review=gate['source_review'];assert re.fullmatch('[0-9a-f]{7,40}',review['commit']) and re.fullmatch('[0-9a-f]{7,40}',review['publication'])
    assert sha(review['path'])==review['sha256']==gate['pins'][review['path']]
    reviewed=read(review['path']);assert reviewed['complete'] and reviewed['pass_all'] and reviewed['execution_allowed']
    assert all(reviewed['pins'][str(p)]==sha(p) for p in direct)
    R['limits']={'cpu':list(resource.getrlimit(resource.RLIMIT_CPU)),'as':list(resource.getrlimit(resource.RLIMIT_AS))}
    save();assert R['limits']=={'cpu':[20,20],'as':[2147483648,2147483648]}
    raw=read(B/'saved/output/results.json');assert raw['complete'] and raw['pass_all']
    events=[json.loads(x) for x in (B/'saved/output/progress.jsonl').read_text().splitlines()]
    assert [x['kind'] for x in events]==['start','pins_checked','training_ledger_checked','evaluation_identities_checked','readme_transcription_checked','complete']
    assert raw['gate_sha256']==sha(B/'saved/gate.json') and raw['protocol_sha256']==sha(B/'saved/protocol.json')
    assert raw['limits']==R['limits']
    event('pins')
    a=read(B/'snapshot/results/a30.json');z=read(B/'snapshot/results/zoo_code_seed0.json')
    assert a['config']['seeds']==[0,1,2,3,4] and a['config']['H']==16 and a['config']['T']==10
    assert a['config']['log2n']==30 and a['config']['eval_n']==100000000
    assert a['n_steps']*1024==a['n_train']==1073741824
    assert len(a['curve'])==64
    for index,row in enumerate(a['curve'],1):
        assert row['step']==index*16384 and len(row['loss'])==len(row['acc'])==5
        assert math.isfinite(row['sec']) and row['sec']>=0
        assert all(math.isfinite(x) and x>=0 for x in row['loss'])
        assert all(math.isfinite(x) and 0<=x<=1 for x in row['acc'])
    ledger=raw['training_ledger']
    assert ledger['config']==a['config'] and ledger['all_curve_records']==a['curve']
    assert ledger['n_train_per_model_reported']==a['n_train'] and ledger['n_steps_reported']==a['n_steps']==a['curve'][-1]['step']
    assert ledger['curve_records']==64 and ledger['model_slots_per_record']==5 and ledger['loss_acc_pairs']==320
    assert ledger['reported_wall_seconds']=={'a30':a['wall_sec'],'zoo_code_seed0':z['wall_sec']}
    assert ledger['curve_seconds_nondecreasing']==all(x['sec']<=y['sec'] for x,y in zip(a['curve'],a['curve'][1:]))
    assert ledger['last_curve_seconds_not_above_reported_wall']==(a['curve'][-1]['sec']<=a['wall_sec'])
    R['training']={'records':64,'pairs':320,'diagnostic_flags_only':True};save();event('training')
    rows={f'a30_seed{k}':v for k,v in a['eval'].items()};rows['zoo_code_seed0']=z['eval']
    assert set(rows)==set(raw['evaluations']) and len(rows)==6
    R['evaluations']={}
    for key in sorted(rows):
        v=rows[key];d=raw['evaluations'][key]
        assert d['stored']==v and v['n']==100000000 and v['seed']==777
        assert all(type(x) in (int,float) and math.isfinite(x) for x in v.values())
        assert all(0<=x<=1 for k,x in v.items() if k not in ['n','seed'])
        residual=common_sum([v[k] for k in ['acc','err_blind','err_23','err_12','err_other']])-1
        blind=Q(v['err_blind'])/Q(v['blind_share'])
        variance=Q(v['acc'])*(1-Q(v['acc']))/v['n'];assert variance>0
        scale=1<<400;root=math.isqrt((variance.numerator*scale*scale)//variance.denominator)
        lo=Q(root,scale);hi=Q(root+1,scale);assert lo*lo<=variance<hi*hi
        assert float(lo)==float(hi);ulp=Q(math.ulp(float(lo)));allow=8*ulp
        nominal=Q(d['nominal_se_decimal100']);pad=hi/Q(10**98)
        error=abs(Q(v['se'])-nominal);saved_error=Q(d['se_absolute_difference_decimal'])
        error_pad=max(error,Q(1,10**200))/Q(10**98)
        R['evaluations'][key]={'residual':pair(residual),'blind_ratio':pair(blind),'variance':pair(variance),
            'sqrt_bracket':[pair(lo),pair(hi)],'rounding_pad':pair(pad),'allowance':pair(allow)};save()
        assert unpair(d['partition_residual_exact'])==residual and abs(residual)<=Q(1,2**50)
        assert unpair(d['blind_error_rate_from_stored_proportions_exact'])==blind
        assert 0<v['blind_share'] and v['err_blind']<=v['blind_share']
        assert abs(Q(v['blind_error_rate'])-blind)<=Q(1,2**48)
        assert unpair(d['nominal_variance_from_stored_accuracy_exact'])==variance
        assert lo-pad<=nominal<=hi+pad
        assert abs(saved_error-error)<=error_pad
        assert float(nominal)==float(lo) and Q(d['se_allowance_decimal'])==allow
        assert max(abs(Q(v['se'])-lo),abs(Q(v['se'])-hi))<=allow
        assert d['missing_integer_counts_not_reconstructed'] is True
        event('evaluation',record=key)
    text=(B/'snapshot/README.md').read_text();expected=[]
    for line in text.splitlines():
        if line.startswith('| zoo code unchanged'):key='zoo_code_seed0'
        elif line.startswith('| trainer.py seed '):key='a30_seed'+re.search('seed (\\d)',line).group(1)
        else:continue
        cells=[x.strip() for x in line.strip('|').split('|')][1:]
        assert len(cells)==5
        for field,printed in zip(['acc','err_blind','err_23','err_12','err_other'],cells):
            if not printed:continue
            value=Q(rows[key][field]);width=Q(1,2*10**len(printed.split('.')[1]))+Q(math.ulp(rows[key][field]))
            expected.append(dict(record=key,field=field,printed=printed,stored_exact=pair(value),absolute_difference_exact=pair(abs(Q(printed)-value)),rounding_allowance_exact=pair(width),pass_=abs(Q(printed)-value)<=width))
    assert len(expected)==len(raw['table_comparisons'])==27
    for wanted,got in zip(expected,raw['table_comparisons']):
        wanted['pass']=wanted.pop('pass_');assert wanted==got and wanted['pass']
    endpoints=re.findall(r'zoo seed spread \((0\.\d+)-(0\.\d+);',text);assert len(set(endpoints))==1
    low,high=map(Q,endpoints[0]);flags={k:low<=Q(v['acc'])<=high for k,v in rows.items()}
    diagnostic=raw['literal_printed_spread_diagnostic']
    assert diagnostic['printed_endpoints']==list(endpoints[0]) and diagnostic['rows_inside_literal_closed_interval']==flags
    assert diagnostic['rounded_endpoints_do_not_define_exact_acceptance'] and diagnostic['not_a_trajectory_distribution_or_runtime_equivalence_test']
    R['table_entries']=27;R['literal_flags']=flags;save();event('table')
    R.update(complete=True,pass_all=True,author_model_calls=0,binary_reads=0,source_pins=protocol['pins'])
    event('complete');save()
if __name__=='__main__':
    try: main()
    except BaseException as e:
        R.update(complete=False,pass_all=False,fatal={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
        if OWN:
            save()
            with (OUT/'progress.jsonl').open('a') as f:
                f.write(json.dumps({'kind':'fatal','message':str(e)})+'\n');f.flush();os.fsync(f.fileno())
        raise
