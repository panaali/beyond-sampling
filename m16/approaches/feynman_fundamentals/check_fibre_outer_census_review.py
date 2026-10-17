"""Independent saved raw ledger/call contraction, with rescue counted once."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter,defaultdict
import ast,hashlib,json,math,time
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/fibre_cold_native/outer_coverage')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def counter(rows):return Counter({tuple(r['key']):r['count'] for r in rows})
def ledger(l):
    total=Counter()
    for c in l['sites'].values():total.update(c)
    assert total==Counter(l['totals'])
    integer=sum(v for k,v in total.items() if k.startswith('integer_control'))
    assert integer==l['integer_control_total'] and sum(total.values())-integer==l['named_numeric_total']
    return total
start=time.process_time();protocol=read(P/'protocol.json');r=read(P/'results.json');s=read(P/'summary.json')
assert r['complete'] and sha(P/'results.json')==s['census_sha256']
assert r['source_sha256']==sha(P/'census.py') and r['protocol_sha256']==sha(P/'protocol.json')
for path,h in protocol['pins'].items():assert sha(path)==h
calls=Counter();memory=Counter();opaque=Counter();totals=Counter();diagnostic=Counter();sites=defaultdict(Counter)
assert len(protocol['components'])==48 and len(set(protocol['components']))==48
for i,path in enumerate(protocol['components']):
    d=read(path);assert d['complete'] and d.get('index',0)==i
    l=d['ledger'];totals.update(ledger(l));opaque.update(counter(l['opaque_calls']))
    for site,c in l['sites'].items():
        sites[site].update(c)
        if "stats['traces'] += 1" in site or "stats['pieces'] += 1" in site:diagnostic.update(c)
    if i!=12:
        calls.update(counter(l['native_calls']));memory.update(counter(l['memory_calls']))
    else:
        rescue_sites=defaultdict(Counter);rescue_types=Counter();rescue_opaque=Counter()
        paths=[protocol['rescue_prepare']]+protocol['rescued_cells'];assert len(paths)==len(set(paths))==145
        for j,p in enumerate(paths):
            child=read(p);assert child['complete']
            if j:assert child['cell']==j-1
            z=child['ledger'];rescue_types.update(ledger(z));calls.update(counter(z['native_calls']));memory.update(counter(z['memory_calls']));rescue_opaque.update(counter(z['opaque_calls']))
            for site,c in z['sites'].items():rescue_sites[site].update(c)
        assert rescue_types==Counter(l['totals']) and dict(rescue_sites)=={k:Counter(v) for k,v in l['sites'].items()}
        assert rescue_opaque==counter(l['opaque_calls']) and 'native_calls' not in l
assert calls==Counter({(z['name'],z['input_shape'],z['output_shape']):z['count'] for z in r['native_calls']})
assert opaque==counter(r['opaque_calls']) and len(memory)==r['memory_shape_keys']
assert totals==Counter(r['raw_categories']) and diagnostic==Counter(r['diagnostic_categories'])
assert dict(sites)=={k:Counter(v) for k,v in r['native_sites'].items()}
adjusted=Counter({k:v-diagnostic[k] for k,v in totals.items()});assert adjusted==Counter(r['categories'])
integer=sum(v for k,v in totals.items() if k.startswith('integer_control'));numeric=sum(totals.values())-integer;diag=sum(diagnostic.values())
assert (numeric,integer,diag)==(1825974971,12576857,159150)==(s['numeric'],s['integer'],s['diagnostic'])
assert numeric+integer-diag==s['named_visible_components']==1838392678
assert s['unchanged_hybrid_named_composition']==numeric+integer-diag+6980+295652==1838695310
byname=defaultdict(Counter);bygroup=defaultdict(Counter);missing=Counter()
for row in r['native_calls']:
    assert row['layout_stride_dtype_history_saved'] is False
    nums={}
    for which in ('input','output'):
        shape=ast.literal_eval(row[which+'_shape'])
        n=math.prod(shape) if isinstance(shape,tuple) and all(type(v) is int and v>=0 for v in shape) else None
        assert n==row[which+'_elements'];nums[which]=n
        if n is None:missing[which+'_shape_keys']+=1;missing[which+'_shape_calls']+=row['count']
    for c in (byname[row['name']],bygroup[row['group']]):
        c['calls']+=row['count'];c['shape_keys']+=1
        for which,n in nums.items():
            if n is not None:c[which+'_elements']+=n*row['count']
assert {k:dict(v) for k,v in byname.items()}==s['by_name']==r['by_name']
assert {k:dict(v) for k,v in bygroup.items()}==s['by_group']==r['by_group']
assert dict(missing)==s['missing_shape_census']==dict(input_shape_keys=17185,input_shape_calls=109459)
matrix=adjusted['matrix_mul']+adjusted['matrix_add'];norm=adjusted['norm_mul']+adjusted['norm_add']+adjusted['special_sqrt']
assert matrix==s['matrix_expression_units']==857804265 and norm==s['norm_expression_units']==123008922
assert s['native_prices_added']==0
out=dict(complete=True,components=48,rescue_cells=144,prepare_once=True,matrix_expression_units=matrix,norm_expression_units=norm,
         norm3_calls=byname['np.linalg.norm']['calls'],dot3_calls=byname['np.dot']['calls'],missing_shape_census=dict(missing),
         unchanged_hybrid=1838695310,summary_sha256=sha(P/'summary.json'),review_cpu=time.process_time()-start,new_estimator_kernel_calls=0,
         scope='All saved raw sites/calls and rescue reconstruction checked. Shapes do not establish per-call layouts/dtypes/runtime targets. No native prices added, no estimate or comparison changed.')
(HERE/'fibre_outer_census_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
