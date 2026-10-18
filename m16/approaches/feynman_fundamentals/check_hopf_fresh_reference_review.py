"""Independent 2,500-block saved hit/state and numerical confidence audit."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
import hashlib,json,math
import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h4_4/m16/approaches/hopf_reference_precision')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=read(P/'protocol.json');s=read(P/'reference_summary.json')
assert s['protocol_sha256']==sha(P/'protocol.json') and s['source_sha256']==sha(P/'reference.py')
assert s['family_size']==p['confidence_family']==25 and s['nominal_joint_coverage']==.95
rows={(r['seed'],r['copy']):r for r in s['rows']};boxes={(r['seed'],r['index']):r for r in s['mc_truth_boxes']}
assert len(rows)==len(boxes)==25
assert set(rows)==set(boxes)=={(a,b) for a in range(5) for b in range(5)}
assert len({r['rng_seed'] for r in p['models']})==25
files=[];draws=0;allcpu=[];checked=[];max_ci_difference=0.
for model in p['models']:
    key=(model['seed'],model['copy']);row=rows[key];box=boxes[key];folder=P/f's{key[0]}_c{key[1]}'
    expected_paths=[folder/f'block{i:03d}.json' for i in range(100)]
    assert set(folder.glob('block*.json'))==set(expected_paths)
    assert row['model_sha256']==model['model_sha256'] and row['rng_seed']==model['rng_seed']
    state=np.random.default_rng(model['rng_seed']).bit_generator.state;hits=0;cpu=[]
    assert len(row['block_records'])==100
    for j,path in enumerate(expected_paths):
        r=read(path);assert r['status']=='complete' and (r['seed'],r['copy'],r['block'])==(*key,j)
        assert type(r['hits']) is int and 0<=r['hits']<=r['n']==1_000_000
        assert r['model_sha256']==model['model_sha256'] and r['rng_seed']==model['rng_seed']
        assert r['protocol_sha256']==s['protocol_sha256'] and r['source_sha256']==s['source_sha256']
        assert r['state_before']==state;state=r['state_after'];assert state['bit_generator']=='PCG64'
        assert state['state']['inc']==r['state_before']['state']['inc']
        assert row['block_records'][j]==dict(block=j,sha256=sha(path))
        hits+=r['hits'];draws+=r['n'];cpu.append(r['cpu_seconds']);files.append(str(path))
    assert hits==row['hits']==box['hits'] and row['n']==box['n']==100_000_000
    assert row['accuracy']==hits/row['n'] and abs(math.fsum(cpu)-row['cpu_seconds'])<3e-12
    assert box['H']==box['T']==4 and 0<box['lower']<hits/row['n']<box['upper']<1
    N=row['n'];alpha=.05/(2*25)
    # Independent binomial-tail inversions; no Bernoulli samples are generated.
    lo=brentq(lambda x:binom.sf(hits-1,N,x)-alpha,0.,hits/N,xtol=5e-15)
    hi=brentq(lambda x:binom.cdf(hits,N,x)-alpha,hits/N,1.,xtol=5e-15)
    delta=max(abs(lo-box['lower']),abs(hi-box['upper']));assert delta<1e-11
    max_ci_difference=max(max_ci_difference,delta)
    checked.append(dict(seed=key[0],copy=key[1],hits=hits,n=N,independent_interval=[lo,hi],saved_interval_difference=delta))
    allcpu.extend(cpu)
assert len(files)==len(set(files))==2500 and draws==s['reference_draws']==2_500_000_000
assert abs(math.fsum(allcpu)-s['reference_cpu_seconds'])<1e-10
assert s['extra_control_evaluations']==318 and s['numerical_beta_not_outward_rounded']
out=dict(complete=True,models=25,blocks=2500,reference_draws=draws,extra_controls=318,
         model_results=checked,max_independent_interval_difference=max_ci_difference,
         reference_cpu_seconds=s['reference_cpu_seconds'],summary_sha256=sha(P/'reference_summary.json'),
         new_random_forward_calls=0,
         scope='Saved exact hit sums and state chaining; fixed 25-family numerical intervals. State continuity is not a replay of random generation. Neither numerical inversion nor floating forward semantics is a real-arithmetic/outward certificate; source-cost comparison remains separate.')
(HERE/'hopf_fresh_reference_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='model_results'},indent=2))
