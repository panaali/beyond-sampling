"""One pinned trained reset-limit sector, using saved exact weights only."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from partition import partition,mass,I
from rational_bounds import serial,accounting

HERE=Path(__file__).resolve().parent
OUT=HERE/'application.json'
assert not OUT.exists(),'Preserve prior application'
sys.set_int_max_str_digits(0)
t0=time.process_time()
gate=json.loads((HERE/'application_gate.json').read_text())
for name,expected in gate['source_sha256'].items():
    assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected,name
controls=json.loads((HERE/'controls.json').read_text())
assert controls['complete'] and len(controls['rows'])==9 and len(controls['rounding_controls'])==4
for name,expected in controls['source_sha256'].items():
    assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected,name
parent=json.loads((HERE/'../application.json').read_text())
assert parent['complete'] and parent['checked_complete_occurrence_rank_pairs']==198
assert len(parent['feasible_face_evidence'])==14
wi=list(map(F,parent['weights']['Wi']))
wh=[list(map(F,row)) for row in parent['weights']['Wh']]
wo=[list(map(F,row)) for row in parent['weights']['Wo']]
assert wi[0]<0<wi[1] and wh[0][0]<0 and wh[1][0]<0
# The first random step resets in the large negative-prefix limit. The
# inserted positive target next gives (0, Wi[1]*u); final input is r*u.
a=[wh[i][1]*wi[1] for i in range(2)]
b=wi
parts=partition(a,b,wo)
result=mass(parts,label=2)
limit=result['joint_correctness_mass']
sector=F(parent['sector_probability'])
assert sector==F(1,12)
event=parent['bounds']['event_probability']
event_upper=F(event['upper'])
assert 0<=event_upper<=sector
original=I(max(F(0),limit.lo-event_upper),min(sector,limit.hi+event_upper))
conditional=original/sector
limit_conditional=limit/sector
representative_count=sum(len(row['representatives']) for row in parts['intervals'])
out={'complete':True,'dependency_gate':gate,'weights':parent['weights'],
     'partition':parts,'mass':result,'sector_probability':sector,
     'inherited_unconditional_disagreement_upper':event_upper,
     'original_joint_correctness_enclosure':original,
     'original_conditional_sector_accuracy_enclosure':conditional,
     'limit_conditional_sector_accuracy_enclosure':limit_conditional,
     'exact_reduced_formula_representative_evaluations':representative_count,
     'exact_reduced_formula_boundary_evaluations':len(parts['boundaries']),
     'checkpoint_loads':0,'original_network_forward_MC_calls':0,
     'accounting':accounting(),'cpu_seconds':time.process_time()-t0,
     'peak_rss_kib_including_imports':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
     'application_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'scope':'Fixed H2T4 seed0, x0<0 and target input u>0 second-largest, target index2. Exact-real interpretation of stored dyadic weights. Sector joint mass only; original enclosure depends on inherited exhaustive shared-face proof. No whole-model accuracy, hardware rounding or sampling-cost claim.'}
with OUT.open('x') as f:json.dump(serial(out),f,indent=2);f.write('\n')
print(json.dumps({'limit_joint':[float(limit.lo),float(limit.hi)],
    'original_joint':[float(original.lo),float(original.hi)],
    'original_conditional_sector_accuracy':[float(conditional.lo),float(conditional.hi)],
    'intervals':len(parts['intervals']),'boundaries':len(parts['boundaries']),
    'reduced_formula_representatives':representative_count,
    'cpu_seconds':out['cpu_seconds'],'accounting':accounting()},indent=2))
