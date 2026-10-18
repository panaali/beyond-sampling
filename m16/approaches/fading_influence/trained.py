"""Frozen768 exact local witnesses; no probability or uniform decay inference."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from itertools import permutations,product
import gzip,json,hashlib,time,resource
from core import integer_model,envelope,witness,pack
HERE=Path(__file__).resolve().parent

def main():
    start=time.process_time();out=HERE/'trained.json';progress=HERE/'trained_progress.jsonl';assert not out.exists() and not progress.exists()
    P=json.loads((HERE/'trained_protocol.json').read_text());assert P['source_gate'] and P['saved_control_review']
    for path,sha in P['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
    controls=json.loads((HERE/'controls.json').read_text());assert controls['complete'] and len(controls['rows'])==12
    result=dict(complete=False,models=[],protocol_sha256=hashlib.sha256((HERE/'trained_protocol.json').read_bytes()).hexdigest(),import_and_prior_cpu=start)
    def log(r):
        with progress.open('a') as f:f.write(json.dumps(pack(r),allow_nan=False)+'\n')
    def save():
        result.update(cpu=time.process_time()-start,process_cpu=time.process_time(),maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        p=out.with_suffix('.tmp');p.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');p.replace(out)
    try:
        # A fixed combinatorial set; no randomness, fitted ordering or selection.
        xs=[list(v*s for v,s in zip(perm,signs)) for perm in permutations((1,2,3,4)) for signs in product((-1,1),repeat=4)]
        assert len(xs)==384 and len(set(map(tuple,xs)))==384
        for name in ('H4T4seed0','H8T4seed0'):
            log(dict(stage='model_attempt',name=name));tick=time.process_time()
            if name=='H4T4seed0':
                d=json.loads(gzip.decompress(Path(P['h4_geometry']).read_bytes()));decode=lambda z:F(int(z[0]),int(z[1]))
                model=(list(map(decode,d['model'][0])),[list(map(decode,r)) for r in d['model'][1]],[list(map(decode,r)) for r in d['model'][2]])
                checkpoint=d['checkpoint_sha256'];assert checkpoint==P['h4_checkpoint_sha256']
            else:
                import torch
                torch.set_num_threads(1);d=torch.load(P['h8_checkpoint'],map_location='cpu',weights_only=True)
                assert set(d)=={'rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'}
                wi=d['rnn.weight_ih_l0'];wh=d['rnn.weight_hh_l0'];wo=d['linear.weight']
                assert tuple(wi.shape)==(8,1) and tuple(wh.shape)==(8,8) and tuple(wo.shape)==(4,8)
                assert torch.isfinite(wi).all() and torch.isfinite(wh).all() and torch.isfinite(wo).all()
                model=(wi[:,0].tolist(),wh.tolist(),wo.tolist());checkpoint=P['pins'][P['h8_checkpoint']]
            m=integer_model(model);assert m['T']==4 and m['H']==(4 if name=='H4T4seed0' else 8)
            loader_cpu=time.process_time()-tick;tick=time.process_time();u=envelope(m);upper_cpu=time.process_time()-tick
            row=dict(name=name,checkpoint_sha256=checkpoint,model=m,envelope=u,loader_cpu=loader_cpu,envelope_cpu=upper_cpu,witnesses=[],complete=False);result['models'].append(row)
            log(dict(stage='model_input',name=name,model=m,envelope=u,loader_cpu=loader_cpu));save()
            for k,x in enumerate(xs):
                log(dict(stage='witness_attempt',name=name,index=k,x=x));tick=time.process_time();w=witness(m,x,u);w['cpu']=time.process_time()-tick;w['index']=k;row['witnesses'].append(w)
                log(dict(stage='witness_result',name=name,witness=w))
                if k%16==15:save()
            strict=[w for w in row['witnesses'] if w['strict']];boundary=[w for w in row['witnesses'] if not w['strict']]
            counts=Counter()
            for w in row['witnesses']:counts.update(w['counts'])
            maximum=[max((w['age_sensitivities'][a] for w in strict),default=None) for a in range(4)]
            old_vs_new=dict(older_greater=0,equal=0,older_smaller=0,newest_zero=0)
            for w in strict:
                old,new=w['age_sensitivities'][3],w['age_sensitivities'][0]
                old_vs_new['older_greater' if old>new else 'equal' if old==new else 'older_smaller']+=1
                old_vs_new['newest_zero']+=int(new==0)
            row['summary']=dict(strict_witnesses=len(strict),unresolved_boundary_witnesses=len(boundary),
                distinct_strict_gate_histories=len({''.join('1' if z else '0' for g in w['history'] for z in g) for w in strict}),
                max_sensitivity_by_age=maximum,zero_sensitivity_counts_by_age=[sum(w['age_sensitivities'][a]==0 for w in strict) for a in range(4)],
                oldest_vs_newest_counts=old_vs_new,integer_named_counts=dict(counts),max_retained_integer_bits=max(w['bits'] for w in row['witnesses']),
                witness_cpu=sum(w['cpu'] for w in row['witnesses']),
                scope='Counts are fixed deterministic witnesses, not Gaussian mass, coverage or merge savings. Boundary witnesses excluded from local derivative summaries.')
            row['complete']=True;log(dict(stage='model_complete',name=name,summary=row['summary']));save()
        assert sum(len(r['witnesses']) for r in result['models'])==768
        result['complete']=True;log(dict(stage='complete',witnesses=768));save()
    except BaseException as e:result['failure']=repr(e);log(dict(stage='failure',exception=repr(e)));save();raise
    print(json.dumps(dict(complete=True,cpu=result['cpu'],summaries=[dict(name=r['name'],strict=r['summary']['strict_witnesses'],boundary=r['summary']['unresolved_boundary_witnesses']) for r in result['models']])))
if __name__=='__main__':main()
