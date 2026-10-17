"""Direct integer recurrence verifies saved whole-gate-cone readout labels."""
from pathlib import Path
from collections import defaultdict,Counter
from fractions import Fraction as F
import gzip,json,time
start=time.process_time()
base=Path('/workspace/work/recover-small_h2_910/m16/approaches/bulk_error_mechanism')
source=Path('/workspace/clean-neurons/m16/approaches/small_ladder_review/rational_geometry/geometry_h4_t4.json.gz')
data=json.loads(gzip.decompress(source.read_bytes()));prior=json.loads((base/'attribution.json').read_text());saved=json.loads((base/'constant_output.json').read_text())
correctness=json.loads((base/'cone_correctness.json').read_text())
conv=lambda x:F(int(x[0]),int(x[1]))
wi=list(map(conv,data['model'][0]));wh=[list(map(conv,r)) for r in data['model'][1]];wo=[list(map(conv,r)) for r in data['model'][2]]
D=max(v.denominator for v in wi+sum(wh,[])+sum(wo,[]))
wi=[int(v*D) for v in wi];wh=[[int(v*D) for v in r] for r in wh];wo=[[int(v*D) for v in r] for r in wo]
rays=defaultdict(set)
for row,key in zip(data['gate_simplices'],prior['sets']['gate']['history_keys']):
    rays[key].update(tuple(map(int,r)) for r in row['integer_rays'])
def forward(x):
    h=[0]*4;scale=1
    for u in x:
        h=[max(0,sum(a*b for a,b in zip(r,h))+v*u*scale) for r,v in zip(wh,wi)]
        scale*=D
    return [sum(a*b for a,b in zip(r,h)) for r in wo]
matches=Counter();witnesses={};calls=0;correctness_counts=Counter()
for key,vertices in rays.items():
    scores=[forward(v) for v in vertices];calls+=len(scores)
    candidates=[]
    for j in range(4):
        okay=True
        for k in range(4):
            if k==j:continue
            differences=[s[j]-s[k] for s in scores]
            if min(differences)<0 or (k<j and max(differences)==0):okay=False
        if okay:candidates.append(j)
    assert candidates==([] if saved['certificates'][key]['constant_label'] is None else [saved['certificates'][key]['constant_label']])
    matches['constant' if candidates else 'varying']+=1
    if candidates:
        j=candidates[0];x=[sum(v[i] for v in vertices) for i in range(4)]
        s=forward(x);assert max(range(4),key=lambda k:s[k])==j
        if j not in witnesses:witnesses[j]={'history':key,'input_integer_ray_sum':list(map(str,x))}
    possible_outputs=[];possible_ranks=[]
    for j in range(4):
        blocked=False;always=0;possible=0
        for k in range(4):
            if k==j:continue
            margin=[s[k]-s[j] for s in scores]
            if min(margin)>=0 and (k<j or max(margin)>0):blocked=True
            rankdiff=[v[k]-v[j] for v in vertices]
            if min(rankdiff)>=0 and max(rankdiff)>0:always+=1
            if max(rankdiff)>0:possible+=1
        if not blocked:possible_outputs.append(j)
        if always<2 and possible:possible_ranks.append(j)
    kind='uniformly_wrong' if set(possible_outputs).isdisjoint(possible_ranks) else 'uniformly_correct' if len(possible_outputs)==len(possible_ranks)==1 and possible_outputs==possible_ranks else 'unresolved'
    expected=correctness['certificates'][key]
    assert possible_outputs==expected['possible_outputs'] and possible_ranks==expected['possible_second_rank'] and kind==expected['kind']
    correctness_counts[kind]+=1
assert len(witnesses)==4
out={'history_count':len(rays),'direct_ray_forwards':calls,'classification':dict(matches),'whole_cone_correctness_counts':dict(correctness_counts),'all_four_recurrence_witnesses':witnesses,'cpu_seconds':time.process_time()-start,
     'scope':'same supplied exact gate geometry, independent direct recurrence; no independent enumeration or mass integration'}
Path(__file__).with_name('constant_cones_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
