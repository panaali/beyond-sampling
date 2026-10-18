"""Saved helper controls, callgraph pins and outer cold-once contraction."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json,math
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/rank_numeric_envelope/outer_review')
C=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_runtime_census')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=read(P/'helper_controls.json');assert a['cases']==len(a['rows'])==630 and a['failures']==0
assert a['source_sha256']==sha(P/'helper_controls.py')
kinds=Counter();max_iterations=0
for r in a['rows']:
    kinds[r['kind']]+=1
    assert r['match'] and r['count']<=r['upper']
    assert all(x==y or math.isnan(float.fromhex(x)) and math.isnan(float.fromhex(y)) for x,y in zip(r['expected'],r['actual']))
    if r['kind']=='dlas2':assert r['upper']==28
    elif r['kind']=='dlascl':
        assert r['upper']==43+3*r['args']['k'] and r['extra']['info']==0
        assert r['extra']['iterations']<=3;max_iterations=max(max_iterations,r['extra']['iterations'])
    elif r['kind']=='dlasrt':assert r['upper']==len(r['args'])*(len(r['args'])-1)//2 and r['extra']['info']==0
    else:assert r['kind']=='ieeeck' and r['args']['abi']=='float32' and r['upper']==30
assert kinds==dict(dlas2=216,dlascl=48,dlasrt=364,ieeeck=2)
for path,h in read(P/'cold_source_pins.json').items():assert sha(Path(path))==h
graph=read(P/'cold_helper_callgraph.json')
for name,v in graph['nodes'].items():
    assert sha(P/f'scipy_{name}_64_.disasm.txt')==v['disassembly_sha256']
    assert not any(any(bad in call for bad in ('dlasq1','dlasq2','dlarre','dbdsqr')) for call in v['calls'])
s=read(P/'contraction.json');modes={}
for name,h in s['pins'].items():assert sha(C/name)==h
for mode in ('exact','recipe108'):
    groups={'geometry':Counter(),'cold_once':Counter()}
    for j in range(8):
        raw=read(C/f'{mode}_{j}.json');cold=Counter()
        for phase in ('native','cold'):
            assert not raw[phase]['overflow']
            for r in raw[phase]['kernels']:
                k=r['kernel']
                if k not in ('dlasq1','dlasq2','dlasq3'):continue
                assert not r['query'] and not r['failures']
                dest=groups['geometry'] if phase=='native' else cold
                if k in ('dlasq1','dlasq2'):dest[k]+=r['calls']
                elif r['m']==4:dest['q3_l4']+=r['calls']
        if j==0:groups['cold_once']=cold
        else:assert cold==groups['cold_once']
    groups['total']=groups['geometry']+groups['cold_once']
    for key,c in groups.items():
        got=s['modes'][mode][key]
        values=dict(q1_calls=c['dlasq1'],q2_calls=c['dlasq2'],q3_entry_l4=c['q3_l4'],
                    q1_exclusive_or_small_helper=28*c['dlasq1'],q1_two_scales=119*c['dlasq2'],
                    q2_exclusive_and_small_helpers=426*c['dlasq2'],q2_split_checks=11*c['q3_l4'])
        values['outer_source_upper']=values['q1_exclusive_or_small_helper']+values['q1_two_scales']+values['q2_exclusive_and_small_helpers']+values['q2_split_checks']
        assert values==got
    modes[mode]=s['modes'][mode]['total']['outer_source_upper']
out=dict(complete=True,helper_controls=dict(kinds),max_scale_iterations=max_iterations,outer_source_upper=modes,
         callgraph_nodes=len(graph['nodes']),new_native_rank_model_calls=0,
         scope='Saved helper parity and source bounds plus raw calls with one cold phase. Small dimensions rely on pinned static values-only cold path and four-column rank invariant, not lastN. Historical binding/BLAS premises and unpriced reductions/native/control/memory remain.')
(HERE/'outer_rank_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
