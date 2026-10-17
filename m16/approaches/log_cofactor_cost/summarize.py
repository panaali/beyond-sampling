"""Contract independently checked existing calls; never evaluate a cone/model."""
from pathlib import Path
import collections,hashlib,json,time
HERE=Path(__file__).resolve().parent
def main():
    start=time.process_time();manifest=json.loads((HERE/'manifest.json').read_text())
    for name,pin in manifest.items():assert hashlib.sha256((HERE/'frozen'/name).read_bytes()).hexdigest()==pin['sha256']
    d=json.loads((HERE/'frozen/full_summary.json').read_text());assert d['pins']['estT_snapshot.py']==manifest['estT_snapshot.py']['sha256']
    assert d['cases']==35098==d['thin']+d['wide'];assert d['inverse_calls']==d['cases']
    assert d['scalar3det_calls']==784*d['wide'];assert d['scalar4det_calls']==d['cases']+128*d['wide']
    assert d['legacy_ops_once']==30144*d['thin']+30536*d['wide']
    b=json.loads((HERE/'baseline.json').read_text());spy=json.loads((HERE/'instrumented.json').read_text());ker=json.loads((HERE/'kernels.json').read_text())
    for z in (spy,ker):assert [{k:v for k,v in r.items() if k not in ('events','spy')} for r in z['rows']]==b['rows']
    for r in ker['rows']:
        n=2 if r['name']=='batch' else 1;events=r['events']
        assert [e['args'] for e in events if e['kernel']=='iamax']==[[3,1,0,0],[2,1,0,0],[1,1,0,0]]*n
        assert [e['args'] for e in events if e['kernel']=='dot']==[[1,3,1,0]]*n
        assert [e['args'] for e in events if e['kernel']=='gemv']==[[3,0,1,1],[2,1,1,1],[1,2,1,1]]*n
        scal=[e['args'] for e in events if e['kernel']=='scal'];assert len(scal)<=2*n
        assert all(a in ([2,1,0,0],[1,1,0,0]) for a in scal)
        assert all(e['kernel'] in ('iamax','dot','gemv','scal','swap') for e in events)
    det3=dict(lu_arithmetic=24,lu_abs_comparison=63,reconstruction=17);assert sum(det3.values())==104
    branches=dict(thin=dict(common_visible=150,extra_visible=38001,det3_calls=0,det4_calls=1,inverse4_calls=1,model_free_runtime=2),wide=dict(common_visible=150,extra_visible=16953,det3_calls=784,det4_calls=129,inverse4_calls=1,model_free_runtime=9))
    for r in branches.values():
        r['visible']=r['common_visible']+r['extra_visible'];r['native_small_subset_envelope']=104*r['det3_calls']+176*r['det4_calls']+290*r['inverse4_calls'];r['hybrid_subtotal']=r['visible']+r['native_small_subset_envelope']
    total_visible=sum(d[k]*v['visible'] for k,v in branches.items());native=dict(determinant3=104*d['scalar3det_calls'],determinant4=176*d['scalar4det_calls'],inverse4=290*d['inverse_calls']);hybrid=total_visible+sum(native.values());constants=sum(d[k]*v['model_free_runtime'] for k,v in branches.items())
    assert hybrid==sum(d[k]*v['hybrid_subtotal'] for k,v in branches.items())
    out=dict(branches=branches,det3_numeric_categories=det3,cases=d['cases'],thin=d['thin'],wide=d['wide'],visible_expression_total=total_visible,native_small_subset_envelopes=native,hybrid_subtotal=hybrid,model_free_runtime_constants=constants,hybrid_plus_runtime_constants=hybrid+constants,old_legacy_mass=d['legacy_ops_once'],hybrid_over_legacy_diagnostic=hybrid/d['legacy_ops_once'],source_sha256=manifest['estT_snapshot.py']['sha256'],full_census_review='4cc3fe39',control_totals=dict(sum((collections.Counter(r['spy']) for r in spy['rows']),collections.Counter())),control_cpu=sum(x['cpu_seconds'] for x in (b,spy,ker)),cpu_seconds=time.process_time()-start,new_mass_model_MC_evaluations=0,complete_native_upper_or_lower=False,sampling_ratio_not_computed=True,scope='Pinned exact35098-cone mass calls only. Hybrid source-expression plus installed small-library category upper; BLAS/NumPy/builtin reduction internals, control/integer/memory, libm internals, geometry and cold setup remain unpriced. Not a full implementation bound or budget-comparable MSE.')
    (HERE/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
