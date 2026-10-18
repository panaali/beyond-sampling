"""Fixed new H4T3 exact capture followed by the previously reviewed chain reduction."""
from pathlib import Path
from fractions import Fraction as F
import gzip, hashlib, importlib.util, json, math, os, resource, sys, time

HERE=Path(__file__).resolve().parent
START=time.process_time()
OUT=HERE/'output'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def verify(pins):
    assert pins
    for p,h in pins.items(): assert sha(p)==h,p
def atomic(p,x,compressed=False):
    b=(json.dumps(x,separators=(',',':'),allow_nan=False)+'\n').encode()
    if compressed: b=gzip.compress(b,mtime=0)
    q=p.with_name(p.name+'.tmp');q.write_bytes(b);q.replace(p)
def load_module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    OUT.mkdir(exist_ok=False)
    result={'complete':False,'pass_all':False,'controls':[],'forward_inputs':0,'rng_calls':0,'new_geometry_captures':0}
    log=(OUT/'progress.jsonl').open('x',buffering=1)
    phase='prerequisites';exact=None;capture_state=None
    def save():
        result.update(cpu_seconds=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if exact is not None:
            result['geometry_work']=dict(exact.counts);result['geometry_observed_bits']=dict(exact.bits)
        atomic(OUT/'results.json',result)
    def event(kind,data):
        log.write(json.dumps({'phase':phase,'kind':kind,'cpu':time.process_time()-START,'data':data},separators=(',',':'))+'\n')
        if time.process_time()-START>17: raise TimeoutError('17 CPU internal cap')
    try:
        p=read(HERE/'protocol.json');g=read(HERE/'gate.json')
        result['runtime']={'invoked':sys.executable,'resolved':str(Path(sys.executable).resolve()),'prefix':sys.prefix,
                           'cpu_limit':resource.getrlimit(resource.RLIMIT_CPU),'as_limit':resource.getrlimit(resource.RLIMIT_AS)}
        save();verify(p['pins']);verify(g['pins'])
        assert sys.executable==p['invoked_python'] and str(Path(sys.executable).resolve())==p['resolved_python']
        assert resource.getrlimit(resource.RLIMIT_CPU)==(20,20) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,2**31)
        assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'))
        for key in ('review_commit','publication'): assert len(g[key])>=8 and all(c in '0123456789abcdef' for c in g[key])
        rv=read(g['review_path']);assert all(rv[k] is True for k in ('complete','pass_all','execution_allowed'));verify(rv['pins'])
        required=[HERE/n for n in ('angle.py','run.py','PROTOCOL.md','protocol.json','launch.sh')]
        for f in required: assert rv['pins'][str(f)]==sha(f)==g['pins'][str(f)]
        assert g['pins'][g['review_path']]==sha(g['review_path'])
        old=read(p['prior_saved_review']);assert old['complete'] is True and old['pass_all'] is True
        # Prior controls and source are immutable; no rerun of those14 cases.
        verify(old['input_pins'])
        result['prior_chain_review']={'sha':sha(p['prior_saved_review']),'controls_reused':8,'guards_reused':6}
        import numpy as np
        result['numpy']={'version':np.__version__,'path':np.__file__};save()
        assert result['numpy']==p['numpy']
        primitive=load_module('t267_reused_chain',HERE.parent/'primitive.py')
        kernel=load_module('t267_trihedral',HERE/'angle.py')
        source=load_module('t267_exact_geometry',p['exact_source'])
        event('prerequisites_passed',{'protocol_sha':sha(HERE/'protocol.json'),'gate_sha':sha(HERE/'gate.json')})
        phase='controls'
        e=[(1,0,0),(0,1,0),(0,0,1)];m=(1,1,0)
        controls=[('octant',e,F(1,8)),('negative',[tuple(-x for x in r) for r in e],F(1,8)),
                  ('permuted',e[::-1],F(1,8)),('scaled',[(3,0,0),(0,5,0),(0,0,7)],F(1,8)),
                  ('split_left',[e[0],m,e[2]],F(1,16)),('split_right',[m,e[1],e[2]],F(1,16))]
        work=primitive.Work()
        for name,rays,truth in controls:
            event('angle_control_attempt',{'name':name})
            det=F(abs(work.det(rays)),math.prod(sum(map(abs,r)) for r in rays))
            row={'rays':rays,'det':primitive.pair(det),'weight':1}
            returned=kernel.mass(row)
            rec={'name':name,'input':row,'result':returned,'exact_truth':primitive.pair(truth)}
            result['controls'].append(rec);save();event('angle_control_result',rec)
            assert abs(F.from_float(float.fromhex(returned['value_hex']))-truth)<=F(1,10**14)
        result['controls_work']=dict(work.counts);result['angle_controls_passed']=True;save()

        phase='capture';event('weight_decode_attempt',{'path':p['weights']})
        clock=time.process_time()
        with np.load(p['weights'],allow_pickle=False) as data:
            assert set(data.files)=={'Wi','Wh','Wo'}
            arrays=tuple(np.array(data[n],copy=True) for n in ('Wi','Wh','Wo'))
        assert [a.shape for a in arrays]==[(4,),(4,4),(3,4)]
        assert all(a.dtype.kind=='f' and a.dtype.itemsize<=8 and np.all(np.isfinite(a)) for a in arrays)
        model=source.dyadic_model(arrays)
        encoded=[[[str(x.numerator),str(x.denominator)] for x in model[0]],
                 [[[str(x.numerator),str(x.denominator)] for x in row] for row in model[1]],
                 [[[str(x.numerator),str(x.denominator)] for x in row] for row in model[2]]]
        atomic(OUT/'model.json',{'weights_sha':sha(p['weights']),'model':encoded,
                                'shapes':[list(a.shape) for a in arrays],'dtypes':[str(a.dtype) for a in arrays]})
        result['decode_cpu']=time.process_time()-clock
        event('weight_decode_result',{'model_sha':sha(OUT/'model.json')})
        exact=source.Exact(seconds=12,max_bits=4096,max_rays=2000)
        event('gate_capture_attempt',{'H':4,'T':3});clock=time.process_time();result['new_geometry_captures']=1;save()
        cells,stages=exact.gates(model)
        def cone(c): return {'rays':c.rays,'tight':c.tight,'constraints':c.constraints}
        gate_rows=[{'cone':cone(c),'map':[[primitive.pair(v) for v in row] for row in A]} for c,A in cells]
        atomic(OUT/'gates.json.gz',{'stages':stages,'cells':gate_rows},True)
        result['gate_capture_cpu']=time.process_time()-clock
        event('gate_capture_result',{'cells':len(cells),'stages':stages,'sha':sha(OUT/'gates.json.gz')})
        assert cells and len(cells)<=20000
        gate_simplices=[];correct_simplices=[];correct_cones=[];clock=time.process_time()
        capture_state={'gate_simplices':gate_simplices,'correct_simplices':correct_simplices,'correct_cones':correct_cones}
        def simplex(item,gate_id):
            rays,det=item
            return {'rays':rays,'expected_det':primitive.pair(det),'gate_id':gate_id}
        for index,(c,A) in enumerate(cells):
            event('gate_geometry_attempt',{'index':index});exact.certify(c)
            for orth in exact.orthants(c):
                exact.certify(orth)
                gate_simplices.extend(simplex(x,index) for x in exact.triangulate(orth))
            good=exact.outcomes(c,A,model)
            atomic(OUT/f'gate_{index:04d}_correct_cones.json.gz',{'gate_id':index,'cones':[cone(x) for x in good]},True)
            for cgood in good:
                exact.certify(cgood)
                correct_cones.append({'gate_id':index,'cone':cone(cgood)})
                for orth in exact.orthants(cgood):
                    exact.certify(orth)
                    correct_simplices.extend(simplex(x,index) for x in exact.triangulate(orth))
            atomic(OUT/'capture.json.gz',capture_state,True)
            save();event('gate_geometry_result',{'index':index,'gate_simplices':len(gate_simplices),'correct_simplices':len(correct_simplices),'correct_cones':len(correct_cones)})
        result['classification_triangulation_recording_cpu']=time.process_time()-clock
        gate_total=sum((F(*map(int,x['expected_det'])) for x in gate_simplices),F())
        correct_total=sum((F(*map(int,x['expected_det'])) for x in correct_simplices),F())
        scaled=correct_total*(1<<128)
        correct_bracket={'bits':128,'lower_units':str(scaled.numerator//scaled.denominator),
                         'upper_units':str(-((-scaled.numerator)//scaled.denominator)),
                         'exact_numerator_bits':correct_total.numerator.bit_length(),
                         'exact_denominator_bits':correct_total.denominator.bit_length()}
        result['capture']={'stages':stages,'gate_cells':len(cells),'gate_simplices':len(gate_simplices),
                           'correct_cones':len(correct_cones),'correct_simplices':len(correct_simplices),
                           'gate_det_total':primitive.pair(gate_total),'correct_det_bracket':correct_bracket,'sha':sha(OUT/'capture.json.gz')}
        save();assert gate_total==8 and 0<correct_total<8

        phase='chain';event('chain_attempt',{'simplices':len(correct_simplices)});clock=time.process_time()
        ans=primitive.reduce_chain([{k:v for k,v in x.items() if k!='gate_id'} for x in correct_simplices],3,primitive.Work(),event)
        atomic(OUT/'chain.json.gz',ans,True);result['chain_cpu']=time.process_time()-clock
        original=primitive.dyadic_totals(ans['originals']);reduced=primitive.dyadic_totals(ans['terms'])
        result['chain']={'sha':sha(OUT/'chain.json.gz'),'original_terms':len(ans['originals']),'boundary_terms':len(ans['terms']),
                         'unique_facets':len(ans['facets']),'cancelled_facets':sum(x['coefficient']==0 for x in ans['facets']),
                         'live_facets':sum(x['coefficient']!=0 for x in ans['facets']),
                         'zero_anchor_facets':sum(x['coefficient']!=0 and int(x['signed_det'][0])==0 for x in ans['facets']),
                         'counts':ans['counts'],'original_totals':original,'reduced_totals':reduced}
        save();event('chain_result',result['chain'])
        assert max(int(original['signed_lower_units']),int(reduced['signed_lower_units']))<=min(int(original['signed_upper_units']),int(reduced['signed_upper_units']))
        phase='integration';result['integrations']={}
        for name,items in [('baseline',ans['originals']),('boundary',ans['terms'])]:
            values=[];clock=time.process_time()
            with (OUT/(name+'.jsonl')).open('x',buffering=1) as stream:
                for i,row in enumerate(items):
                    event('angle_attempt',{'method':name,'index':i});returned=kernel.mass(row)
                    stream.write(json.dumps({'index':i,'result':returned},separators=(',',':'))+'\n')
                    values.append(float.fromhex(returned['value_hex']))
                    event('angle_result',{'method':name,'index':i,'value_hex':returned['value_hex']})
            result['integrations'][name]={'calls':len(items),'value':math.fsum(values),'value_hex':math.fsum(values).hex(),
                                         'absolute_sum':math.fsum(map(abs,values)),'cpu':time.process_time()-clock,'sha':sha(OUT/(name+'.jsonl')),
                                         'named_kernel_categories_per_call':45,'rational_coordinate_conversions_per_call':9}
            save()
        a,b=result['integrations']['baseline'],result['integrations']['boundary']
        result['comparison']={'difference':b['value']-a['value'],'term_ratio':b['calls']/a['calls'],
                              'integration_cpu_saved':a['cpu']-b['cpu'],'floating_evaluation_certified':False,
                              'shared_new_discovery_cpu':result['decode_cpu']+result['gate_capture_cpu']+result['classification_triangulation_recording_cpu'],
                              'chain_cpu_includes_recording':True}
        result['pass_all']=True;event('complete',result['comparison']);result['complete']=True;save()
    except BaseException as exc:
        if capture_state is not None: atomic(OUT/'capture.json.gz',capture_state,True)
        result.update(complete=False,pass_all=False,failure={'phase':phase,'type':type(exc).__name__,'message':str(exc)})
        save();log.write(json.dumps({'phase':phase,'kind':'fatal','data':result['failure']})+'\n');raise
    finally: log.close()


if __name__=='__main__':main()
