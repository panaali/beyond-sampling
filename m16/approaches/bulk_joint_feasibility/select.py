"""Freeze largest unresolved gate histories by input mass, never correctness."""
import time
START=time.process_time()
import gzip,hashlib,json,resource,sys
from collections import defaultdict
from fractions import Fraction as F
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;PRIOR=P.parent/'bulk_error_mechanism';sys.path.insert(0,str(PRIOR));import attribute as A
stored=A.SOURCE.read_bytes();data=json.loads(gzip.decompress(stored));assign=json.loads((PRIOR/'attribution.json').read_text());certs=json.loads((PRIOR/'cone_correctness.json').read_text())['certificates'];masses=defaultdict(float);bounds=defaultdict(float)
for row,key in zip(data['gate_simplices'],assign['sets']['gate']['history_keys']):
    rays=[]
    for raw in row['integer_rays']:
        v=list(map(int,raw));den=sum(abs(x) for x in v);rays.append([float(F(x,den)) for x in v])
    mass,bound=A.V.mass(np.array(rays).T,20);masses[key]+=mass;bounds[key]+=bound
assert abs(sum(masses.values())-1)<1e-12
eligible=[k for k in masses if certs[k]['kind']=='unresolved'];selected=sorted(eligible,key=lambda k:(-masses[k],k))[:10]
result=dict(complete=True,selection_rule='ten largest prior unresolved gate masses, history-bitstring tiebreak; no correct/error data used',selected=[dict(history=k,mass=masses[k],conditional_quadrature_bound=bounds[k]) for k in selected],selected_gate_mass=sum(masses[k] for k in selected),unselected_gate_mass=1-sum(masses[k] for k in selected),all_mass=sum(masses.values()),all_mass_conditional_bound=sum(bounds.values()),source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [A.SOURCE,PRIOR/'attribution.json',PRIOR/'cone_correctness.json',A.MASS]},cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
assert result['cpu_seconds_including_imports']<20
(P/'selection.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
