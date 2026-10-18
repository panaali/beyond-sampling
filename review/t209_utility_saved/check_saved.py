"""Independent saved-only utility audit; no candidate or geometry/mass calls."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib, importlib.util, json, math, subprocess, time, traceback

H=Path(__file__).resolve().parent
A=Path('/workspace/work/recover-small_h4_4/review/cap_saved_utility')
REPO=H.parents[1]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def outward(q, saved):
    nearest=float(q); changed=int(F(nearest)<q)
    value=math.nextafter(nearest,math.inf) if changed else nearest
    assert saved['bound']==str(q) and saved['outward'].hex()==value.hex()==saved['outward_hex']
    assert saved['conversion']==dict(fallback=None,nextafter=changed)
    assert q<=F(value)<=1

def main():
    start=time.process_time(); pin=read(H/'pins.json')
    for p,h in pin['pins'].items(): assert sha(p)==h,p
    proto=read(A/'protocol.json'); gate=read(A/'gate.json'); raw=read(A/'results.json')
    for p,h in {**proto['pins'],**gate['pins']}.items(): assert sha(p)==h,p
    assert gate['complete'] and raw['complete']
    assert gate['protocol_sha256']==raw['protocol_sha256']==sha(A/'protocol.json')
    assert raw['gate_sha256']==sha(A/'gate.json')
    for name,commit in [('results.json','c9a03d9d'),('progress.jsonl','c9a03d9d'),('gate.json','c5152b7b')]:
        assert subprocess.check_output(['git','show',f'{commit}:review/cap_saved_utility/{name}'],cwd=REPO)==(A/name).read_bytes()
    for c in ['c9a03d9d','c5152b7b','1a1ea51f','9a95a036']:
        subprocess.run(['git','merge-base','--is-ancestor',c,'e35f8896'],cwd=REPO,check=True)
    for c,target in [('c5152b7b','c9a03d9d'),('9d5faa2b','e35f8896')]:
        subprocess.run(['git','merge-base','--is-ancestor',c,target],cwd=REPO,check=True)
    # Import only our previously reviewed independent rational reconstruction.
    helper=Path(pin['own_helper'])
    spec=importlib.util.spec_from_file_location('own_t200_review',helper)
    own=importlib.util.module_from_spec(spec); spec.loader.exec_module(own)
    import numpy as np
    source=Path(proto['data_root']); oldp=read(source/'pilot_protocol.json')
    old=read(source/'pilot.json'); certs=read(source/'pilot_certificates.json')
    capture=read(source/'capture.json'); prior=read(proto['certificate_review'])
    assert all(d['complete'] for d in [old,certs,capture,prior])
    assert prior['raw_sha256']==sha(source/'pilot.json')
    assert prior['certificates_sha256']==old['certificates_sha256']==sha(source/'pilot_certificates.json')
    assert prior['protocol_sha256']==old['protocol_sha256']==certs['protocol_sha256']==sha(source/'pilot_protocol.json')
    indices=[i*37738//63 for i in range(64)]
    assert proto['indices']==oldp['indices']==old['indices']==indices
    assert [r['index'] for r in raw['rows']]==[r['index'] for r in certs['rows']]==[r['index'] for r in old['rows']]==indices
    assert [r['index'] for r in capture['parents']]==list(range(96))
    arrays=[]
    for p in capture['parents']:
        path=source/'parents'/p['archive']; assert sha(path)==p['sha256']
        with np.load(path,allow_pickle=False) as z:
            assert z.files==['rays']; arr=z['rays']
        assert arr.dtype==np.dtype('float64') and arr.shape==(p['N'],4,4) and np.isfinite(arr).all()
        arrays.append(arr)
    allr=np.concatenate(arrays)
    assert allr.shape==(37739,4,4) and allr.nbytes==raw['loaded_ray_bytes']
    assert hashlib.sha256(allr.tobytes()).hexdigest()==proto['ordered_ray_sha256']==oldp['ordered_ray_sha256']
    progress=[json.loads(s) for s in (A/'progress.jsonl').read_text().splitlines()]
    assert len(progress)==130
    assert progress[0]==dict(stage='started',gate_sha256=sha(A/'gate.json'))
    assert progress[-1]==dict(stage='completed',result_sha256=sha(A/'results.json'))
    counts=dict(binary64_to_fraction=16,rational_add=52,rational_mul=46,rational_compare=22,rational_div=4,rational_sub=1)
    counts.update({k:1 for k in ['integer_shift','integer_divmod','integer_compare','integer_isqrt','integer_square_compare','rational_construct','rational_square_compare']})
    total=Counter(); diagnostics={'cap':[],'determinant':[]}; times=[]
    for i,(r,c,o) in enumerate(zip(raw['rows'],certs['rows'],old['rows'],strict=True)):
        index=indices[i]; arr=allr[index]
        assert r['rays_hex']==[[float(v).hex() for v in row] for row in arr]
        assert r['ray_sha256']==hashlib.sha256(arr.tobytes()).hexdigest()
        assert r['certificate']==c['certificate'] and c['independent_validation']['passed']
        assert r['original_reference']==o['reference'] and r['original_reference_bound']==o['reference_bound']
        assert progress[2*i+1]==dict(stage='row_started',index=index)
        assert progress[2*i+2]==dict(stage='row_completed',index=index,cap_fallback=None)
        R=[[F(float(v)) for v in row] for row in arr]
        axis,q,delta,b,peak=own.geometry(R)
        cap=r['cap']; assert cap['fallback'] is None
        assert cap['details']==dict(dimension=4,rays=4,q=str(q),delta=str(delta),axis=list(map(str,axis)))
        assert cap['ledger']==dict(counts=counts,counted_steps=148,max_fraction_bits=peak,bit_limit=8192,step_limit=128000)
        total.update(counts); outward(b,cap)
        # Reuse independently verified KKT/determinant certificate, no new solve.
        a=F(c['certificate']['a']); det=F(c['certificate']['det']); assert a>0 and det!=0
        db=min(F(1),abs(det)/(108*a*a)); outward(db,r['determinant_upper'])
        ref=F(o['reference']['value']); error=F(o['reference_bound']['total_bound']); assert ref>0 and error>=0
        for label,arm,bound in [('cap',cap,b),('determinant',r['determinant_upper'],db)]:
            ratio=F(arm['outward'])/ref; disagreement=ref-error>bound
            assert r[label+'_diagnostic']==dict(rational_to_stored_reference=str(bound/ref),outward_to_stored_reference=str(ratio),conditional_reference_disagreement=disagreement)
            diagnostics[label].append((index,F(arm['outward']),ref,ratio,disagreement))
        times.extend([r['cap_cpu_seconds'],r['determinant_contraction_cpu_seconds']])
    summaries={}
    assert proto['thresholds']==[str(F(1,10**k)) for k in range(4,9)]
    for label,ds in diagnostics.items():
        ratios=sorted(d[3] for d in ds); thresholds=[]
        for text in proto['thresholds']:
            selected=[d for d in ds if d[1]<=F(text)]
            thresholds.append(dict(threshold=text,indices=[d[0] for d in selected],count=len(selected),outward_bound_sum=str(sum((d[1] for d in selected),F(0))),stored_reference_sum=str(sum((d[2] for d in selected),F(0)))))
        summaries[label]=dict(minimum_ratio=str(ratios[0]),median_ratio=str((ratios[31]+ratios[32])/2),maximum_ratio=str(ratios[-1]),conditional_reference_disagreement_indices=[d[0] for d in ds if d[4]],thresholds=thresholds)
    assert raw['summary']==summaries and raw['cap_macro_counts']==dict(total) and raw['cap_fallbacks']=={'none':64}
    assert raw['new_cap_calls']==raw['new_determinant_contractions']==64
    assert raw['new_KKT_solves']==raw['new_geometry_mass_model_reference_calls']==0
    assert raw['certificate_construction_inherited']==dict(review_commit='b5bb78d0',operations=prior['recorded_construction_operations'],validation_operations=prior['recorded_validation_operations'],peak_bits=prior['recorded_peak_rational_bits'])
    assert all(math.isfinite(t) and t>=0 for t in times+[raw['cpu_seconds'],raw['load_hash_cpu_seconds']])
    assert sum(times)+raw['load_hash_cpu_seconds']<=raw['cpu_seconds']+1e-12
    return dict(status='PASS_SAVED_ONLY',rows=64,progress_records=130,archives=96,source_pins=len(proto['pins']),summaries=summaries,cap_macro_counts=dict(total),inherited_certificate_accounting=raw['certificate_construction_inherited'],author_cpu=raw['cpu_seconds'],review_cpu=time.process_time()-start,author_imports=0,cap_geometry_mass_model_reference_calls=0,scope='Exact saved represented-cone bounds and diagnostic contractions. Reuses published KKT verification; no new solve, mass/reference, complete pruning budget, production cost or unsampled-cone conclusion.',pins=pin['pins'])

if __name__=='__main__':
    assert not (H/'results.json').exists() and not (H/'failure.txt').exists()
    try: result=main()
    except BaseException:
        with (H/'failure.txt').open('x') as f:f.write(traceback.format_exc())
        raise
    with (H/'results.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],rows=64,cpu=result['review_cpu'])))
