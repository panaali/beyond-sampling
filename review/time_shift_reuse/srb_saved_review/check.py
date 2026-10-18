"""Independent AST and saved-ledger audit. No Q, NumPy, SciPy or author imports."""
import ast,json,hashlib,struct,math,time,resource,traceback
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent
A=Path('/workspace/work/recover-small_h2_34/review/srb_cost_review/q_controls')
OBS='_srb_observer_ledger'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def shape_size(shape):return math.prod(shape)
def broadcast(shapes):
    result=[]
    for i in range(max(map(len,shapes),default=0)):
        dims=[s[-1-i] if i<len(s) else 1 for s in shapes];nonones={x for x in dims if x!=1};assert len(nonones)<=1
        result.append(next(iter(nonones),1))
    return result[::-1]
def encoded(rows,shape):
    values=[float(v) for v in rows];return dict(shape=shape,dtype='float64',hex=[v.hex() for v in values],bytes_sha256=hashlib.sha256(struct.pack('<'+'d'*len(values),*values)).hexdigest())
def restore(node):
    # Rebuild generic AST fields recursively, recognizing observer syntax only.
    if isinstance(node,list):return [r for x in node if (r:=restore(x)) is not None]
    if not isinstance(node,ast.AST):return node
    if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id==OBS and node.value.func.attr=='interval':return None
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id==OBS:
        method=node.func.attr;a=node.args
        if method=='call':return ast.Call(func=restore(a[1]),args=restore(a[2:]),keywords=restore(node.keywords))
        if method=='aug_rhs':return restore(a[3])
        op=getattr(ast,a[1].value)()
        if method=='binary':return ast.BinOp(left=restore(a[2]),op=op,right=restore(a[3]))
        if method=='unary':return ast.UnaryOp(op=op,operand=restore(a[2]))
        if method=='compare':return ast.Compare(left=restore(a[2]),ops=[op],comparators=[restore(a[3])])
        raise AssertionError(method)
    return type(node)(**{k:restore(v) for k,v in ast.iter_fields(node)})
def kind(n):
    if isinstance(n,ast.Call):return 'call'
    if isinstance(n,ast.AugAssign):return 'inplace_'+type(n.op).__name__
    if isinstance(n,(ast.BinOp,ast.UnaryOp)):return type(n.op).__name__
    if isinstance(n,ast.Compare):assert len(n.ops)==1;return type(n.ops[0]).__name__

def check():
    pins=read(HERE/'pins.json')
    for p,h in pins['pins'].items():assert sha(p)==h,p
    p=read(A/'protocol.json');gate=read(A/'gate.json');d=read(A/'results.json');trees=read(A/'trees.json');fixtures=read(A/'fixtures.json')
    for path,h in {**p['pins'],**gate['pins']}.items():assert sha(path)==h,path
    assert gate['complete'] and gate['protocol_sha256']==sha(A/'protocol.json')==d['protocol_sha256'] and d['gate_sha256']==sha(A/'gate.json')
    src=ast.parse(Path(p['original_source']).read_text());funcs=[x for x in src.body if isinstance(x,ast.FunctionDef) and x.name in ('h9','Q')];original=ast.Module(body=funcs,type_ignores=[])
    assert ast.dump(ast.parse(trees['original']))==ast.dump(original)
    assert ast.dump(restore(ast.parse(trees['observed'])))==ast.dump(original)
    assert hashlib.sha256(trees['original'].encode()).hexdigest()==d['extracted_source_sha256'] and hashlib.sha256(trees['observed'].encode()).hexdigest()==d['transformed_source_sha256']
    sites={}
    for f in funcs:
        for n in ast.walk(f):
            k=kind(n)
            if k:sites.setdefault((f'{f.name}:{n.lineno}:{n.col_offset}',k),[]).append(n)
    assert len(fixtures)==len(d['rows'])==6 and d['complete'] and d['paired_evaluations']==21 and d['fixture_count']==6
    assert d['all_bit_parity'] and d['all_warning_parity'] and d['all_source_count_parity']
    assert not d['complete_native_cost'] and not d['full_accuracy_certificate'] and d['new_trained_model_checkpoint_MC_draw_calls']==0 and d['scope']==p['scope']
    progress=[dict(stage='started')];reports=[];total_events=total_calls=0
    for case,row,number in zip(fixtures,d['rows'],[3,3,4,3,4,4],strict=True):
        assert case['name']==row['name'];assert len(row['runs'])==number==len(case['x'])+1
        assert len(case['Wi'])==16 and len(case['Wh'])==16 and all(len(r)==16 for r in case['Wh']) and len(case['Wo'])==10 and all(len(r)==16 for r in case['Wo'])
        progress.append(dict(stage='fixture_started',name=case['name']))
        for i,run in enumerate(row['runs']):
            name='batch' if i==0 else f'row_{i-1}';xx=case['x'] if i==0 else [case['x'][i-1]];M=len(xx)
            assert run['name']==name and all(len(r)==9 for r in xx);assert run['input']==encoded([x for r in xx for x in r],[M,9])
            progress.append(dict(stage='call_started',fixture=case['name'],name=name))
            assert run['original']==run['observed'] and run['original']==encoded([float.fromhex(x) for x in run['original']['hex']],[M])
            assert run['original_warnings']==run['observed_warnings']==sorted(run['original_warnings'])
            assert all(len(w)==2 and all(isinstance(s,str) for s in w) for w in run['original_warnings'])
            assert run['bit_parity'] and run['warning_parity'] and run['count_parity']
            ledger=run['ledger'];intervals=ledger['intervals'];assert len(intervals)==19
            for k,it in enumerate(intervals):assert it['k']==k and len(it['mask'])==M and all(type(x)is bool for x in it['mask']) and it['active']==any(it['mask'])
            J=sum(x['active'] for x in intervals);counts=Counter();frequency=Counter();call_shapes=Counter()
            for e in ledger['events']:
                total_events+=1;site=e['site'];k=e['kind'];nodes=sites[(site,k)];node=nodes[0];frequency[(site,k)]+=1
                output=e['output'];inputs=e['inputs']
                if k=='call':
                    total_calls+=1;fun=node.func;fname=fun.attr if isinstance(fun,ast.Attribute) else fun.id;fname='ndtr' if fname=='cdf' else fname
                    assert e['name']==('absolute' if fname=='abs' else fname) and set(e['kwargs'])=={q.arg for q in node.keywords}
                    call_shapes[json.dumps({z:e[z] for z in ('site','name','bound','inputs','kwargs','output')},sort_keys=True)]+=1
                    if fname in ('where','maximum','minimum'):assert output['shape']==broadcast([x['shape'] for x in inputs])
                    if fname in ('abs','isnan','isinf','ndtr'):assert output['shape']==inputs[0]['shape']
                    if fname in ('argsort','sort'):assert output['shape']==inputs[0]['shape']
                    if fname in ('max','min'):assert e['bound']['shape']==[M,10] and output['shape']==[M]
                    if fname=='any':assert (e['bound']['shape'],output['shape']) in [([M],[]),([M,10],[M])]
                    if fname=='h9':assert output['shape']==[M,16] and inputs[1]['shape']==[M,9]
                    if fname=='len':assert output==dict(shape=[],dtype='int')
                    continue
                units=shape_size(output['shape']);assert units==e['units']
                dt=output['dtype'];cat='bool' if dt in ('bool','bool_') else 'float' if dt.startswith(('float','complex')) else 'integer_control';assert cat==e['category']
                if k=='MatMult':
                    left,right=inputs;assert len(left['shape'])==len(right['shape'])==2
                    m,inner=left['shape'];inner2,n=right['shape'];assert inner==inner2 and output['shape']==[m,n]
                    assert (left['shape'],right['shape']) in [([M,16],[16,16]),([M,16],[16,10])]
                    counts['dense_2mac']+=2*m*inner*n;counts['dense_dot_expression']+=m*n*(2*inner-1)
                else:
                    if k not in ('Not',):assert output['shape']==broadcast([x['shape'] for x in inputs])
                    counts[cat+('_array' if output['shape'] else '_scalar')]+=units
            for (site,k),nodes in sites.items():
                fn,line,_=site.split(':');line=int(line)
                expected=9 if fn=='h9' and line==18 else (1 if fn=='h9' else (0 if line in (30,31) else (19 if line in (46,47,48) else J if 50<=line<=64 else 1)))
                expected*=len(nodes)
                assert frequency[(site,k)]==expected,(site,k,frequency[(site,k)],expected)
            assert dict(counts)==ledger['counts']
            formulas=dict(float_array=320*M+9*M+J*M*110,dense_2mac=5120*M+640*J*M,dense_dot_expression=4960*M+620*J*M)
            assert run['expected_source_counts']==formulas and all(counts[k]==v for k,v in formulas.items())
            assert counts['bool_array']==16+19*M+J*M*72 and counts['float_scalar']==1+J and counts['bool_scalar']==19 and counts['integer_control_scalar']==30
            progress.append(dict(stage='call_completed',fixture=case['name'],name=name,J=J))
            reports.append(dict(fixture=case['name'],name=name,active_intervals=J,events=len(ledger['events']),counts=dict(counts),call_shape_counts=dict(call_shapes),warnings=run['original_warnings']))
        assert row['batch_vs_single_hex_equal']==(row['runs'][0]['original']['hex']==[r['original']['hex'][0] for r in row['runs'][1:]])
        if case['name']=='zero_readout_preserved_tie':assert row['preserved_tie_defect'] and all(float.fromhex(x)==0 for x in row['runs'][0]['original']['hex'])
        progress.append(dict(stage='fixture_completed',name=case['name']))
    progress.append(dict(stage='completed',results_sha256=sha(A/'results.json')))
    assert progress==[json.loads(x) for x in (A/'progress.jsonl').read_text().splitlines()]
    return dict(pass_all=True,fixtures=6,pairs=21,author_Q_executions=42,event_count=total_events,call_count=total_calls,progress_records=len(progress),runs=reports,author_cpu_seconds=d['cpu_seconds'],reviewer_author_Q_numpy_scipy_model_mass_calls=0,scope='Saved parity and named expression ledger only; no independent numerical correctness or complete native cost claim')
def main():
    out=HERE/'results_v3.json';assert not out.exists();resource.setrlimit(resource.RLIMIT_CPU,(19,20));start=time.process_time();r=dict(complete=False,checker_sha256=sha(__file__),pins_sha256=sha(HERE/'pins.json'));out.write_text(json.dumps(r,indent=2)+'\n')
    try:r.update(check(),complete=True)
    except BaseException as e:r.update(failure=repr(e),traceback=traceback.format_exc());raise
    finally:r['review_cpu_seconds']=time.process_time()-start;out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('complete','fixtures','pairs','event_count','call_count','review_cpu_seconds')}))
if __name__=='__main__':main()
