"""Rank witnesses for exactly128 previously audited represented hulls."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,hashlib,time,sys,resource
from primitive import witness,pack
H=Path(__file__).resolve().parent

def rat(x):return F(int(x[0]),int(x[1]))

def main():
    out=H/'application.json';progress=H/'application_progress.jsonl';assert not out.exists() and not progress.exists()
    p=json.loads((H/'application_protocol.json').read_text())
    assert p['source_gate']!='pending' and p['control_review']!='pending'
    start=time.process_time();result=dict(complete=False,protocol_sha256=hashlib.sha256((H/'application_protocol.json').read_bytes()).hexdigest())
    def log(x):
        with progress.open('a') as f:f.write(json.dumps(pack(x),allow_nan=False)+'\n')
    def cap():
        if time.process_time()-start>17:raise RuntimeError('17CPU phase cap')
    log(dict(stage='phase_attempt',protocol_sha256=result['protocol_sha256']))
    try:
        for path,sha in p['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
        assert str(Path(sys.executable).resolve())==p['runtime_python']
        controls=json.loads((H/'controls.json').read_text());assert controls['complete'] and controls['all_pass'] and controls['cases']==12
        saved_review=json.loads(Path(p['saved64_review']).read_text());assert saved_review['complete'] and saved_review['pass_all']
        old=json.loads(Path(p['trained']).read_text());assert old['complete'] and [m['H'] for m in old['models']]==[4,8]
        rows=[];counts=Counter();det_counts=Counter();rational_decodes=0;summary={}
        for model in old['models']:
            assert len(model['rows'])==32
            for row in model['rows']:
                assert row['H']==model['H'] and len(row['result']['domains'])==2
                for side,dom in enumerate(row['result']['domains']):
                    cap();ident=dict(H=row['H'],truth=row['truth'],rank=row['rank'],pair_index=row['pair_index'],side=side,occurrence_id=row['a' if side==0 else 'b']['occurrence_id'])
                    log(dict(stage='attempt',identity=ident))
                    # No canonicalizer or source network replay: these exact keys/signs were independently audited.
                    linear,hinges=dom['canonical']['key']
                    zero=not hinges and not any(rat(x) for r in linear for x in r)
                    assert zero==dom['certified']
                    rays=[tuple(map(rat,r)) for r in dom['rays']]
                    ans=witness(rays);full=ans['status']=='full_rank_witness'
                    status='certified_equal_inherited' if zero else ('certified_unequal_on_represented_hull' if full else 'unknown_rank_deficient')
                    saved=dict(identity=ident,inherited_status=dom['status'],canonical_nonzero=not zero,
                               selected_original_rays=[dom['rays'][i] for i in ans['selected_indices']],result=ans,status=status)
                    rows.append(saved);counts.update(ans['elimination_counts']);rational_decodes+=ans['rational_decodes']
                    if ans['determinant'] is not None:det_counts.update(ans['determinant']['counts'])
                    log(dict(stage='result',row=saved));cap()
            subset=[r for r in rows if r['identity']['H']==model['H']]
            assert len(subset)==64
            summary[str(model['H'])]=dict(hulls=64,statuses=dict(Counter(r['status'] for r in subset)),ranks=dict(Counter(r['result']['rank'] for r in subset)))
        ids=[tuple(r['identity'][k] for k in ('H','truth','rank','side')) for r in rows]
        assert len(rows)==128 and len(set(ids))==128 and set(ids)=={(h,t,k,s) for h in (4,8) for t in range(4) for k in range(8) for s in (0,1)}
        result.update(complete=True,rows=rows,summary=summary,elimination_arithmetic=dict(counts),determinant_arithmetic=dict(det_counts),rational_decodes=rational_decodes,
                      cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      inherited_trained_sha256=p['pins'][p['trained']],new_canonical_geometry_forward_mass_reference_calls=0,scope=p['scope'])
        log(dict(stage='complete'));out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');print(json.dumps(dict(complete=True,cpu=result['cpu_seconds'],summary=summary)))
    except BaseException as e:
        result.update(failure=repr(e),cpu_seconds=time.process_time()-start);log(dict(stage='failure',message=repr(e)));out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');raise
if __name__=='__main__':main()
