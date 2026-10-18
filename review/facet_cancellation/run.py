"""Once-only finite controls and fixed archived H4T4 cancellation discriminator."""
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from fractions import Fraction as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE/'output'
START = time.process_time()
from primitive import Work, reduce_chain, dyadic_totals, pair, archived_integer
from fixtures import fixtures


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, obj, compressed=False):
    data = (json.dumps(obj, separators=(',', ':'), allow_nan=False)+'\n').encode()
    if compressed:
        data = gzip.compress(data, mtime=0)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_bytes(data)
    tmp.replace(path)


def main():
    OUT.mkdir(exist_ok=False)
    result = {'complete': False, 'pass_all': False, 'controls': [], 'guards': [],
              'model_calls': 0, 'geometry_enumerations': 0, 'rng_calls': 0}
    progress = (OUT/'progress.jsonl').open('x', buffering=1)
    phase = 'prerequisites'
    def event(kind, data):
        progress.write(json.dumps({'phase': phase, 'kind': kind, 'cpu': time.process_time()-START,
                                   'data': data}, separators=(',', ':'), allow_nan=False)+'\n')
        if time.process_time()-START > 100:
            raise TimeoutError('100 CPU second internal cap')
    def save():
        result['cpu_seconds'] = time.process_time()-START
        result['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        atomic(OUT/'results.json', result)
    try:
        p = json.loads((HERE/'protocol.json').read_text())
        g = json.loads((HERE/'gate.json').read_text())
        result['runtime'] = {'invoked': sys.executable, 'resolved': str(Path(sys.executable).resolve()),
                             'prefix': sys.prefix, 'cpu_limit': resource.getrlimit(resource.RLIMIT_CPU),
                             'as_limit': resource.getrlimit(resource.RLIMIT_AS)}
        save()
        assert str(Path(sys.executable).resolve()) == p['resolved_python']
        assert sys.executable == p['invoked_python']
        assert all(0 < z <= 110 for z in resource.getrlimit(resource.RLIMIT_CPU))
        assert all(0 < z <= 2147483648 for z in resource.getrlimit(resource.RLIMIT_AS))
        assert all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'))
        for mapping in (p['pins'], g['pins']):
            for path, digest in mapping.items():
                assert sha(path) == digest, path
        for k in ('review_commit', 'publication'):
            assert len(g[k]) >= 8 and all(c in '0123456789abcdef' for c in g[k])
        review = json.loads(Path(g['review_path']).read_text())
        assert review['complete'] and review['pass_all'] and review['execution_allowed']
        for path, digest in review['pins'].items():
            assert sha(path) == digest, path
        for name in ('primitive.py','fixtures.py','run.py','PROTOCOL.md','protocol.json'):
            path = str(HERE/name)
            assert review['pins'][path] == sha(path) == g['pins'][path]
        assert g['pins'][g['review_path']] == sha(g['review_path'])
        import numpy as np
        result['numpy'] = {'version': np.__version__, 'path': np.__file__}
        save()
        assert np.__version__ == p['numpy_version'] and np.__file__ == p['numpy_path']
        event('prerequisites_passed', {'protocol_sha': sha(HERE/'protocol.json'), 'gate_sha': sha(HERE/'gate.json')})

        phase = 'controls'
        def moments(items, d):
            answer = [F(0)]*(1+d+d*(d+1)//2)
            for row in items:
                r = [[F(x, sum(map(abs, ray))) for x in ray] for ray in row['rays']]
                sums = [sum(v[j] for v in r) for j in range(d)]
                value = [F(1)] + [s/d for s in sums]
                value += [(sums[j]*sums[k]+sum(v[j]*v[k] for v in r))/(d*(d+1))
                          for j in range(d) for k in range(j, d)]
                factor = F(*map(int,row['det']))*row['weight']
                answer = [a+factor*b for a,b in zip(answer,value,strict=True)]
            return [pair(a) for a in answer]
        control_rows, guard_rows = fixtures()
        assert len(control_rows) == 8 and len(guard_rows) == 6
        for name,d,rows,live,unique,cancelled in control_rows:
            event('control_attempt', {'name': name})
            ans = reduce_chain(rows,d,Work())
            atomic(OUT/(name+'.json.gz'),ans,True)
            a,b = moments(ans['originals'],d),moments(ans['terms'],d)
            record = {'name':name,'d':d,'archive_sha':sha(OUT/(name+'.json.gz')),
                      'original_moments':a,'reduced_moments':b,
                      'live':sum(x['coefficient'] != 0 for x in ans['facets']),
                      'unique':len(ans['facets']), 'cancelled':sum(x['coefficient']==0 for x in ans['facets'])}
            result['controls'].append(record)
            save()
            event('control_result',record)
            assert (record['live'],record['unique'],record['cancelled']) == (live,unique,cancelled)
            assert a == b
            assert F(*map(int,a[0])) == (2 if name=='duplicate4' else 1)
        for name,rows,d in guard_rows:
            event('guard_attempt', {'name':name})
            guard_work = Work()
            try:
                reduce_chain(rows,d,guard_work)
            except ValueError as exc:
                record={'name':name,'rejected':True,'message':str(exc)}
            else:
                record={'name':name,'rejected':False}
            record['counts'] = dict(guard_work.counts)
            result['guards'].append(record)
            save()
            event('guard_result',record)
            assert record['rejected']
        result['controls_passed'] = True
        save()

        phase = 'saved_complex'
        clock = time.process_time()
        event('archive_decode_attempt', {'path':p['archive']})
        payload = json.loads(gzip.decompress(Path(p['archive']).read_bytes()))
        assert payload['checkpoint_sha256'] == p['checkpoint_sha256']
        assert len(payload['correct_simplices']) == 5058 and len(payload['gate_simplices']) == 3853
        # Decoded model metadata are retained only for provenance; no recurrence or weights operation.
        result['input'] = {'checkpoint_sha256': payload['checkpoint_sha256'],
                           'correct_simplices':5058,'gate_simplices':3853,'archive_sha':sha(p['archive'])}
        rows = [{'rays':[[archived_integer(x) for x in ray] for ray in row['integer_rays']],
                 'expected_det':row['normalized_absolute_determinant']} for row in payload['correct_simplices']]
        result['archive_decode_conversion_cpu'] = time.process_time()-clock
        clock = time.process_time()
        ans = reduce_chain(rows,4,Work(),event)
        atomic(OUT/'chain.json.gz',ans,True)  # returned chain durable before cap/assertions
        result['chain_cpu'] = time.process_time()-clock
        clock = time.process_time()
        original_totals,reduced_totals = dyadic_totals(ans['originals']),dyadic_totals(ans['terms'])
        result['chain'] = {'archive_sha':sha(OUT/'chain.json.gz'),'source_simplices':len(rows),
                           'unique_facets':len(ans['facets']),
                           'cancelled_facets':sum(x['coefficient']==0 for x in ans['facets']),
                           'remaining_facets':sum(x['coefficient']!=0 for x in ans['facets']),
                           'zero_anchor_terms':sum(x['coefficient']!=0 and int(x['signed_det'][0])==0 for x in ans['facets']),
                           'integration_terms':len(ans['terms']), 'counts':ans['counts'],
                           'original_totals':original_totals,'reduced_totals':reduced_totals}
        result['dyadic_accounting_cpu'] = time.process_time()-clock
        save()
        event('chain_result',result['chain'])
        assert max(int(original_totals['signed_lower_units']),int(reduced_totals['signed_lower_units'])) <= min(int(original_totals['signed_upper_units']),int(reduced_totals['signed_upper_units']))

        phase = 'quadrature'
        event('rule_attempt', {'order':20})
        clock = time.process_time()
        x,w = np.polynomial.legendre.leggauss(20)
        x,w = (x+1)/2,w/2
        u,v,z = np.meshgrid(x,x,x,indexing='ij')
        wu,wv,wz = np.meshgrid(w,w,w,indexing='ij')
        y = np.stack((u,(1-u)*v,(1-u)*(1-v)*z,(1-u)*(1-v)*(1-z)),axis=-1).reshape(-1,4)
        weights = (wu*wv*wz*(1-u)**2*(1-v)).ravel()
        np.savez(OUT/'rule.npz', y=y,weights=weights,x=x,w=w)
        result['rule']={'order':20,'nodes':len(weights),'sha':sha(OUT/'rule.npz'),'setup_cpu':time.process_time()-clock}
        save()
        event('rule_result',result['rule'])
        assert len(weights) == 8000 and np.all(weights > 0)
        result['integration'] = {}
        for label,items,totals in (('baseline',ans['originals'],original_totals),('boundary',ans['terms'],reduced_totals)):
            values=[]
            clock=time.process_time()
            with (OUT/(label+'_integrals.jsonl')).open('x',buffering=1) as stream:
                for index,row in enumerate(items):
                    event('mass_attempt',{'method':label,'index':index})
                    columns = np.array([[float(F(x,sum(map(abs,ray)))) for x in ray] for ray in row['rays']]).T
                    points = y @ columns.T
                    q = np.sum(points*points,axis=1)
                    integral = float(np.dot(weights,1/(q*q)))
                    determinant = float(F(*map(int,row['det'])))
                    value = row['weight']*(determinant/(2*np.pi**2)*integral)
                    rec={'index':index,'weight':row['weight'],'determinant_hex':determinant.hex(),
                         'kernel_integral_hex':integral.hex(),'value_hex':value.hex(),
                         'q_min_hex':float(np.min(q)).hex(),'q_max_hex':float(np.max(q)).hex()}
                    stream.write(json.dumps(rec,separators=(',',':'))+'\n')
                    values.append(value)
                    event('mass_result',{'method':label,'index':index,'value_hex':value.hex()})
                    assert math.isfinite(value) and float(np.min(q)) >= .25-1e-12 and float(np.max(q)) <= 1+1e-12
            tv = F(int(totals['absolute_upper_units']),1<<totals['bits'])
            bound = tv*F(249,54*3**19)  # rigorous real-rule upper using pi > 3; excludes floating evaluation
            result['integration'][label]={'calls':len(items),'nodes':8000*len(items),
                'value':math.fsum(values),'value_hex':math.fsum(values).hex(),'sum_abs':math.fsum(map(abs,values)),
                'ideal_truncation_bound_upper':pair(bound),'ideal_truncation_bound_float':float(bound),
                'cpu_seconds':time.process_time()-clock,'rows_sha':sha(OUT/(label+'_integrals.jsonl')),
                'named_arithmetic_per_node':{'matmul_mul':16,'matmul_add':12,'norm_square':4,'norm_add':3,'q_square':1,'reciprocal':1,'weighted_mul':1,'reduction_add_allowance':1}}
            save()
        b,c=result['integration']['baseline'],result['integration']['boundary']
        result['comparison']={'boundary_minus_baseline':c['value']-b['value'],
                              'combined_ideal_bound':b['ideal_truncation_bound_float']+c['ideal_truncation_bound_float'],
                              'node_ratio':c['nodes']/b['nodes'],'integration_cpu_ratio':c['cpu_seconds']/b['cpu_seconds'],
                              'cold_boundary_components_cpu':result['archive_decode_conversion_cpu']+result['chain_cpu']+result['dyadic_accounting_cpu']+result['rule']['setup_cpu']+c['cpu_seconds'],
                              'component_timer_excludes_imports_prerequisites_controls_and_some_serialization':True,
                              'inherited_enumeration_not_included_in_incremental_timer':True,
                              'floating_evaluation_certified':False}
        result['pass_all']=True
        event('complete',result['comparison'])
        result['complete']=True
        save()
    except BaseException as exc:
        result['complete']=False
        result['pass_all']=False
        result['failure']={'phase':phase,'type':type(exc).__name__,'message':str(exc)}
        save()
        progress.write(json.dumps({'phase':phase,'kind':'fatal','data':result['failure']})+'\n')
        raise
    finally:
        progress.close()


if __name__ == '__main__':
    main()
