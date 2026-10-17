"""One frozen stored-weight h3 mean; no network/truth/reference regeneration."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import sys,json,hashlib,time,resource
import numpy as np
import mpmath as mp
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from primitive import prepare,evaluate,pack,mpq
from network import build,check_maps
from network_controls import reference

def main():
    out=HERE/'result.json';progress=HERE/'progress.jsonl';geom=HERE/'geometry.json'
    assert not any(p.exists() for p in (out,progress,geom))
    P=json.loads((HERE/'protocol.json').read_text())
    assert P['control_saved_review'] and P['runner_source_review'], 'Independent review gates required'
    for path,sha in P['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
    start=time.process_time();rows=[];counts=Counter();bits=[0,0];candidate_cpu=0.;reference_cpu=0.
    def record(row):
        with progress.open('a') as f:f.write(json.dumps(pack(row),allow_nan=False)+'\n')
    def write(path,obj):
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(pack(obj),indent=2,allow_nan=False)+'\n');tmp.replace(path)
    record(dict(stage='model_attempt',model='H16T4seed3'))
    try:
        old=json.loads(Path(P['weights_json']).read_text());assert old['complete'] and len(old['rows'])==1
        model=old['rows'][0];assert model['name']=='H16T4seed3' and model['passed']
        wi=list(map(F,model['wi']));wh=[list(map(F,r)) for r in model['wh']]
        assert len(wi)==16 and len(wh)==16 and all(len(r)==16 for r in wh)
        t=time.process_time();cert=build(wi,wh);build_cpu=time.process_time()-t
        write(geom,dict(certificate=cert,build_cpu=build_cpu,complete=False))
        assert len(cert['sectors'])<=34
        t=time.process_time();maps=check_maps(cert);validation_cpu=time.process_time()-t
        write(geom,dict(certificate=cert,build_cpu=build_cpu,map_controls=maps,validation_cpu=validation_cpu,complete=True))
        record(dict(stage='geometry_complete',sectors=len(cert['sectors']),map_records=len(maps['rows'])))
        for k,sector in enumerate(cert['sectors']):
            for j,v in enumerate(sector['h3_rows']):
                record(dict(stage='cell_attempt',sector=k,unit=j))
                t=time.process_time();c=prepare(sector['normals'],v);v160=evaluate(c,160);v200=evaluate(c,200);ct=time.process_time()-t
                t=time.process_time();ref=reference(sector,v);rt=time.process_time()-t
                with mp.workdps(220):
                    x=mp.mpf(v200['value']);y=mp.mpf(ref['value']);err=abs(x-y)
                    cell=dict(sector=k,unit=j,facet=c,value160=v160,value200=v200,reference=ref,
                              absolute_error=mp.nstr(err,100),reference_pass=bool(err<=mp.mpf('1e-60')+abs(y)*mp.mpf('1e-50')),
                              precision_pass=bool(abs(x-mp.mpf(v160['value']))<=mp.mpf('1e-100')*(1+abs(x))),
                              positivity_pass=bool(x>=-mp.mpf('1e-100')),candidate_cpu=ct,reference_cpu=rt)
                rows.append(cell);counts.update(c['counts']);bits=[max(a,b) for a,b in zip(bits,c['bits'])]
                candidate_cpu+=ct;reference_cpu+=rt;record(dict(stage='cell_result',cell=cell))
                assert cell['reference_pass'] and cell['precision_pass'] and cell['positivity_pass'],(k,j)
        with mp.workdps(220):
            mu=[sum((mp.mpf(c['value200']['value']) for c in rows if c['unit']==j),mp.mpf(0)) for j in range(16)]
            rm=[sum((mp.mpf(c['reference']['value']) for c in rows if c['unit']==j),mp.mpf(0)) for j in range(16)]
            mass=sum((mp.atan2(mpq(s['hi'][1]),mpq(s['hi'][0]))-mp.atan2(mpq(s['lo'][1]),mpq(s['lo'][0])) for s in cert['sectors']),mp.mpf(0))/(2*mp.pi)
            assert abs(mass-1)<mp.mpf('1e-100')
            # Existing sampled reference only; no random input generation or recurrence.
            with np.load(P['sample_reference'],allow_pickle=False) as z:
                samples=z['mean'].copy();blocks=z['block_means'].copy();se=z['mean_block_se'].copy()
            assert samples.shape==(4,16) and blocks.shape==(8,4,16) and se.shape==(4,16)
            assert np.isfinite(samples).all() and np.isfinite(blocks).all() and np.isfinite(se).all()
            to_mp=lambda values:[mpq(F(float(v))) for v in values]
            sample=to_mp(samples[2]);bref=[to_mp(blocks[b,2]) for b in range(8)]
            closure=json.loads(Path(P['closure_result']).read_text());mixture=json.loads(Path(P['mixture_result']).read_text())
            assert closure['complete'] and mixture['complete']
            assert closure['reference_sha256']==P['pins'][P['sample_reference']]
            assert closure['weight_sha256']==P['original_weight_sha256']
            cr=[r for r in closure['rows'] if r['step']==3];mr=[r for r in mixture['rows'] if r['step']==3]
            assert len(cr)==len(mr)==1
            methods={name:to_mp(r['mean']) for name,r in cr[0]['methods'].items()}
            assert set(methods)=={'inherited_gauss','exact_h2_then_inherited_gauss'}
            methods['retained_sector_gaussian']=to_mp(mr[0]['mean']);methods['exact_facet_h3']=mu
            assert all(len(values)==16 for values in methods.values())
            assert len(mu)==len(rm)==len(sample)==16 and all(len(b)==16 for b in bref)
            def relative(a,b):return mp.sqrt(sum((x-y)**2 for x,y in zip(a,b))/sum(y*y for y in b))
            diagnostic={name:dict(mean=[mp.nstr(x,200) for x in values],relative_rms_to_exact_mean=mp.nstr(relative(values,mu),100),
                                   relative_rms_to_saved_sample=mp.nstr(relative(values,sample),100),
                                   block_relative_errors=[mp.nstr(relative(values,b),100) for b in bref]) for name,values in methods.items()}
            summary=dict(mean=[mp.nstr(x,200) for x in mu],reference_mean=[mp.nstr(x,100) for x in rm],sector_mass=mp.nstr(mass,100),
                         diagnostic=diagnostic,saved_mean_block_se=se[2].tolist(),max_cell_absolute_error=max((c['absolute_error'] for c in rows),key=mp.mpf))
        result=dict(complete=True,all_checks_pass=True,model='H16T4seed3',rows=rows,summary=summary,
                    counts=dict(facet_exact_named=dict(counts),facet_retained_result_bits=bits,network=cert['exact_named_counts'],network_extra=cert['extra_named_counts'],validation=maps['exact_named_counts']),
                    reference_evaluations=sum(c['reference']['evaluations'] for c in rows),candidate_cpu=candidate_cpu,reference_cpu=reference_cpu,
                    build_cpu=build_cpu,validation_cpu=validation_cpu,cpu=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    protocol_sha256=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),scope=P['scope'])
        write(out,result);record(dict(stage='model_complete',cells=len(rows),cpu=result['cpu']))
        print(json.dumps(dict(complete=True,cells=len(rows),cpu=result['cpu'],diagnostic={n:{k:v for k,v in d.items() if k not in ['mean','block_relative_errors']} for n,d in diagnostic.items()})))
    except BaseException as exc:
        failure=dict(complete=False,failure=repr(exc),rows=rows,cpu=time.process_time()-start)
        write(out,failure);record(dict(stage='failure',exception=repr(exc)));raise
if __name__=='__main__':main()
