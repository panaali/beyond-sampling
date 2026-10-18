"""T231 fixed saved-mass arithmetic. Never imports an author module."""
import hashlib, json, math, os, re, sys, time, traceback
from fractions import Fraction as F
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'output'
START = time.process_time()

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def put(p, obj):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, p)

def rat(x):
    x = F(x)
    return {'numerator': str(x.numerator), 'denominator': str(x.denominator), 'float': float(x)}

def event(kind, **kw):
    with (OUT / 'progress.jsonl').open('a') as f:
        f.write(json.dumps({'event': kind, 'cpu': time.process_time()-START, **kw})+'\n')

def pincheck(pins):
    assert isinstance(pins, dict) and pins
    for path, sha in pins.items():
        assert re.fullmatch('[0-9a-f]{64}', sha) and digest(path) == sha, path

def integer_masses(m):
    pairs = [float(x).as_integer_ratio() for x in m]
    power = max(d.bit_length()-1 for _, d in pairs)
    units = [n << (power-(d.bit_length()-1)) for n, d in pairs]
    return units, power

def summarize_units(units, power, deltas, reported):
    w = sorted(units, reverse=True)
    total = sum(w)
    if not w or total <= 0 or w[-1] < 0:
        return {'valid_nonnegative_total': False}
    requests = []
    for ds in deltas:
        # Decimal scientific criterion and exact value of Python's rounded subtraction.
        for kind, threshold in [('decimal_delta', 1-F(ds)),
                                ('rounded_one_minus_binary_delta', F.from_float(1-float(ds)))]:
            requests.append((threshold, ds, kind))
    requests.sort()
    answers = {ds:{} for ds in deltas}
    i = 0; cumulative = 0
    for threshold, ds, kind in requests:
        target = threshold * total
        while i < len(w) and cumulative < target:
            cumulative += w[i]; i += 1
        assert i > 0 and cumulative >= target and cumulative-w[i-1] < target
        answers[ds][kind] = {
            'N': i, 'reported_N': int(reported[ds]), 'matches_reported': i == int(reported[ds]),
            'previous_shortfall_relative_total': rat((target-(cumulative-w[i-1]))/total),
            'selected_excess_relative_total': rat((cumulative-target)/total)}
    sumsq = sum(x*x for x in w)
    denominator = 1 << power
    return {'valid_nonnegative_total': True, 'total': rat(F(total, denominator)),
            'total_minus_one': rat(F(total, denominator)-1), 'minimum': rat(F(w[-1], denominator)),
            'maximum': rat(F(w[0], denominator)), 'zero_count': sum(x == 0 for x in w),
            'inverse_simpson': rat(F(total*total, sumsq)), 'coverage': answers}

def run_one(spec):
    begin = time.process_time()
    path = ROOT / spec['archive']
    row = {'key': spec['key'], 'archive_sha256': digest(path), 'report_sha256':digest(ROOT/spec['report']),
           'kind':spec['kind'], 'k':spec['k']}
    with np.load(path, allow_pickle=False) as z:
        row['keys'] = sorted(z.files)
        arrays = {key: z[key] for key in z.files}
    row['arrays'] = {key:{'shape':list(a.shape), 'dtype':str(a.dtype)} for key,a in arrays.items()}
    expected = ['event','gate','mass'] if spec['kind']=='width' else ['complete','mass','patterns']
    assert row['keys']==expected, row
    m = arrays['mass']; assert m.ndim==1 and m.dtype==np.dtype('float64') and len(m)>0
    row['mass_count'] = len(m)
    row['nonfinite'] = int(np.count_nonzero(~np.isfinite(m)))
    row['negative'] = int(np.count_nonzero(m < 0))
    row['zero'] = int(np.count_nonzero(m == 0))
    report = read(ROOT/spec['report'])
    if spec['kind']=='width':
        report = report[spec['key']]
        g = arrays['gate']; e = arrays['event']
        assert g.ndim==1 and len(g)==len(m) and g.dtype.kind in 'iu'
        assert e.ndim==1 and len(e)==len(m) and e.dtype.kind in 'US'
        assert int(g.min())==0
        count = int(g.max())+1
        row['dense_gate_ids'] = bool(np.array_equal(np.unique(g),np.arange(count)))
        row['regions'] = count
        row['reported_regions'] = report['regions']
        row['complete'] = None # Archive has no completeness field.
        expected_counts = {ds:report['N'][f'{float(ds):.0e}'] for ds in spec['deltas']}
    else:
        pats = arrays['patterns']; complete = arrays['complete']
        assert pats.dtype==np.dtype('uint8') and pats.shape==(len(m),2*spec['k'])
        assert complete.shape==() and complete.dtype==np.dtype('bool')
        row['unique_patterns'] = int(len(np.unique(pats,axis=0)))
        row['regions'] = len(m); row['reported_regions'] = report['regions']
        row['complete'] = bool(complete)
        row['reported_complete'] = report['complete']
        row['subtrees'] = report['subtrees'];row['subtrees_done'] = report['subtrees_done']
        expected_counts = {ds:report['N_delta'][f'{float(ds):g}'] for ds in spec['deltas']}
    # Persist schema and adverse mass observations before arithmetic; no clipping or repair.
    put(OUT/(spec['key']+'_schema.json'), row)
    if row['nonfinite'] or row['negative']:
        row['arithmetic_skipped'] = 'Nonfinite or negative saved mass; no probability ranking asserted.'
        row['cpu'] = time.process_time()-begin
        return row
    units, power = integer_masses(m)
    if spec['kind']=='width':
        grouped = [0]*row['regions']
        for idx, val in zip(g.tolist(), units, strict=True): grouped[idx] += val
        units = grouped
        # Separate reporting-arithmetic reproduction, no author module or mass evaluation.
        fw = np.sort(np.bincount(g, weights=m))[::-1]
        cum = np.cumsum(fw)/fw.sum()
        source_counts = {ds:int(np.searchsorted(cum,1-float(ds))+1) for ds in spec['deltas']}
        float_total = float(fw.sum())
    else:
        fw = np.sort(m)[::-1];cum = np.cumsum(fw)
        source_counts = {ds:int(np.searchsorted(cum,cum[-1]*(1-float(ds)))+1) for ds in spec['deltas']}
        float_total = float(fw.sum())
    row['exact'] = summarize_units(units,power,spec['deltas'],expected_counts)
    row['source_arithmetic_counts'] = source_counts
    row['source_arithmetic_matches'] = {ds:source_counts[ds]==expected_counts[ds] for ds in spec['deltas']}
    row['source_sorted_sum'] = float_total
    row['reported_sum_minus_one'] = report.get('mass_sum_minus_1',report.get('leafsum_minus_1'))
    row['source_sum_minus_one'] = float_total-1
    row['reported_sum_discrepancy'] = (float_total-1)-row['reported_sum_minus_one']
    if spec['kind']=='gpu':
        row['source_inverse_simpson'] = float(1/np.sum((fw/fw.sum())**2))
        row['reported_inverse_simpson'] = report['mass_eff_regions']
        row['inverse_simpson_discrepancy'] = row['source_inverse_simpson']-report['mass_eff_regions']
    row['cpu'] = time.process_time()-begin
    return row

def main():
    assert not OUT.exists(), 'Never overwrite an attempted audit.'
    protocol=read(HERE/'protocol.json');gate=read(HERE/'gate.json')
    assert gate.get('status')=='ready' and re.fullmatch('[0-9a-f]{8,40}',gate['review_publication'])
    reviewer=Path(gate['source_review']);rv=read(reviewer)
    assert rv.get('status')=='passed' and rv.get('source_review_complete') is True
    mandatory={str(HERE/'check.py'),str(HERE/'protocol.json'),str(ROOT/'manifest.json'),str(reviewer)}
    assert mandatory <= set(gate['pins'])
    assert mandatory-{str(reviewer)} <= set(rv['pins'])
    pincheck(gate['pins']);pincheck(rv['pins']);pincheck(protocol['pins'])
    manifest=read(ROOT/'manifest.json')
    for rec in manifest['records']:
        assert digest(ROOT/rec['snapshot'])==rec['sha256']
    assert len(protocol['datasets'])==12 and len({s['key'] for s in protocol['datasets']})==12
    OUT.mkdir();event('start', datasets=12)
    result={'status':'running','pins':gate['pins'],'protocol_sha256':digest(HERE/'protocol.json'),
            'manifest_sha256':digest(ROOT/'manifest.json'),'rows':[], 'scope':protocol['scope']}
    put(OUT/'results.json',result)
    try:
        for spec in protocol['datasets']:
            assert time.process_time()-START < 16, 'Graceful CPU guard; preserve partial audit.'
            event('dataset_start',key=spec['key'])
            row=run_one(spec);result['rows'].append(row)
            put(OUT/(spec['key']+'.json'),row);put(OUT/'results.json',result)
            event('dataset_complete',key=spec['key'])
        by={r['key']:r for r in result['rows']}
        growth=[]
        for family, depths in [('target',[2,3,4]),('zoo16_4',[2,3,4]),('zoo16_5',[2,3,4]),('gpu',[3,4,5])]:
            for lo,hi in zip(depths,depths[1:]):
                a,b=by[f'{family}_k{lo}'],by[f'{family}_k{hi}']
                if 'exact' not in a or 'exact' not in b: continue
                curves={ds:rat(F(b['exact']['coverage'][ds]['decimal_delta']['N'],
                                      a['exact']['coverage'][ds]['decimal_delta']['N'])) for ds in a['exact']['coverage']}
                growth.append({'family':family,'k_from':lo,'k_to':hi,'regions':rat(F(b['regions'],a['regions'])), 'coverage':curves})
        result['growth']=growth
        result['status']='complete';result['cpu']=time.process_time()-START
        result['report_count_discrepancies']=sum(not v for r in result['rows'] for v in r.get('source_arithmetic_matches',{}).values())
        result['adverse_mass_datasets']=[r['key'] for r in result['rows'] if r['nonfinite'] or r['negative']]
        put(OUT/'results.json',result);event('complete',rows=len(result['rows']))
    except BaseException as exc:
        result['status']='failed';result['error']=repr(exc);result['traceback']=traceback.format_exc()
        put(OUT/'failure.json',result);event('failed',error=repr(exc));raise

if __name__=='__main__': main()
