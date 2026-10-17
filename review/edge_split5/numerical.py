"""Fixed child Duffy rules and reviewed ideal-rule bounds; no adaptive choices."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import importlib.util,json,math,resource,time,warnings
import mpmath as mp
import numpy as np
from geometry import pinned,primitive,sha
P=Path(__file__).resolve().parent

def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def reference(R,dps):
    # Exactly the stored 2D wedge times an independent positive 3D orthant.
    A=[[F(float(x)) for x in row] for row in R]
    assert all(A[i][j]==F(i==j) for i in range(2,5) for j in range(5))
    assert all(A[i][j]==0 for i in range(2) for j in range(2,5))
    cross=A[0][0]*A[1][1]-A[0][1]*A[1][0]
    dot=A[0][0]*A[1][0]+A[0][1]*A[1][1]
    assert cross>0
    with mp.workdps(dps):
        x=mp.mpf(cross.numerator)/cross.denominator;y=mp.mpf(dot.numerator)/dot.denominator
        return mp.nstr(mp.atan2(x,y)/(16*mp.pi),dps)

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(105,110))
    dest=P/'numerical.json';assert not dest.exists()
    start=time.process_time();p=pinned();g=json.loads((P/'geometry.json').read_text())
    assert g['complete'] and g['protocol_sha256']==sha(P/'protocol.json') and g['source_sha256']==sha(P/'geometry.py')
    assert [r['name'] for r in g['rows']]==p['fixture_names']
    prior=json.loads((Path(p['prior'])/'results.json').read_text())
    parent={r['name']:r for r in prior['fixtures']}
    radial=primitive(p);bound=radial.Bound();loader=load_file('split5_source_loader',Path(p['prior'])/'loader.py');m=loader.load()
    out=dict(complete=False,protocol_sha256=sha(P/'protocol.json'),geometry_sha256=sha(P/'geometry.json'),
        source_sha256=sha(__file__),extracted_ast_sha256=m.extracted_ast_sha256,
        numpy_version=np.__version__,mpmath_version=mp.__version__,children=[],setup=[],rows=[],groups=[],
        bound_cold_counts=bound.cold_counts,
        scope='Fixed binary child cones; ideal-rule bound excludes floating nodes, evaluation, determinant and reference roundoff; source proxy is not native cost.')
    def save():
        out.update(cpu=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        dest.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    try:
        for row in g['rows']:
            if row['status']!='accepted':continue
            refs=[]
            for side,hh in enumerate(row['children_hex']):
                R=np.array([[float.fromhex(x) for x in rr] for rr in hh]);r80=reference(R,80);r120=reference(R,120)
                with mp.workdps(140):assert abs(mp.mpf(r80)-mp.mpf(r120))<mp.mpf('1e-70')
                t=time.process_time();cert=radial.certificate(hh)
                out['children'].append(dict(fixture=row['name'],side=side,rays_hex=hh,
                    reference80=r80,reference120=r120,certificate=cert,setup_cpu=time.process_time()-t))
                refs.append(r120)
            with mp.workdps(140):assert abs(sum(map(mp.mpf,refs))-mp.mpf(parent[row['name']]['reference120']))<mp.mpf('1e-70')
        for order in p['orders']:
            t=time.process_time();m.set_rules5(order,128);m.THIN_DET5[0]=math.inf
            out['setup'].append(dict(order=order,thin_nodes=len(m._DU5[0][0]),log_nodes=len(m._LOG5[0][0]),cpu=time.process_time()-t,
                scope='Two rule generations and array setup timed, not included in source proxy. Unused log rule retained unchanged.'))
            for child in out['children']:
                R=np.array([[float.fromhex(x) for x in rr] for rr in child['rays_hex']]);ops=loader.Ops();m.STATS['simplices']=0
                record=dict(fixture=child['fixture'],side=child['side'],order=order);out['rows'].append(record);t=time.process_time()
                try:
                    with warnings.catch_warnings(record=True) as ws:
                        warnings.simplefilter('always');value=float(m._orth5(R.copy(),ops));record['warnings']=[str(w.message) for w in ws]
                    record.update(returned_hex=value.hex(),finite=math.isfinite(value))
                    if record['finite']:
                        with mp.workdps(140):
                            error=mp.mpf(value)-mp.mpf(child['reference120'])
                            record.update(value=value,error=mp.nstr(error,70),absolute_error=mp.nstr(abs(error),70),
                                relative_error=mp.nstr(abs(error/mp.mpf(child['reference120'])),70),probability_range=0<=value<=1)
                    record['ideal_rule_bound']=bound.one(child['certificate'],order)
                except Exception as e:record['exception']=repr(e)
                finally:
                    record.update(cpu=time.process_time()-t,source_proxy_ops=ops.n,source_simplices=m.STATS['simplices']);save()
            for row in g['rows']:
                selected=[r for r in out['rows'] if r['fixture']==row['name'] and r['order']==order]
                gr=dict(fixture=row['name'],order=order,complete=len(selected)==2 and all('exception' not in r and r.get('finite') for r in selected))
                if gr['complete']:
                    with mp.workdps(150):
                        total=sum(mp.mpf(r['value']) for r in selected);ref=mp.mpf(parent[row['name']]['reference120']);err=total-ref
                        # Decimal bound endpoints added in exact rational arithmetic, not binary64.
                        ub=sum(F(r['ideal_rule_bound']['upper_bound']) for r in selected)
                        old=next(r for r in prior['rows'] if r['fixture']==row['name'] and r['arm']=='duffy'+str(order))
                        gr.update(total_exact_float_sum=mp.nstr(total,100),reference=mp.nstr(ref,120),error=mp.nstr(err,70),
                            absolute_error=mp.nstr(abs(err),70),relative_error=mp.nstr(abs(err/ref),70),
                            absolute_pass=abs(err)<=mp.mpf(p['absolute_tolerance']),relative_pass=abs(err/ref)<=mp.mpf(p['relative_tolerance']),
                            bound_sum_fraction=str(ub),bound_sum=mp.nstr(mp.mpf(ub.numerator)/ub.denominator,70),bound_qualifies=ub<=F(p['absolute_tolerance']),
                            original_unsplit_error=old.get('error'),source_proxy_ops=sum(r['source_proxy_ops'] for r in selected),
                            nodes=2*order**4,assembly_additions=1)
                out['groups'].append(gr)
            save()
        assert len(out['rows'])==2*g['accepted']*len(p['orders'])
        out['complete']=True
    except BaseException as e:out['failure']=repr(e);raise
    finally:save();print(json.dumps({k:out[k] for k in ('complete','cpu','max_rss_kib')}))
if __name__=='__main__':main()
