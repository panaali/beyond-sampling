"""Frozen unbounded negative-prefix/positive-target crossing projection."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time,importlib.util
import torch
from project import project,witness,union,complement
from controls import serial
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('crossing_faces',HERE.parent/'face_strip_continuity/faces.py');faces=importlib.util.module_from_spec(spec);spec.loader.exec_module(faces)

def main():
    start=time.process_time()
    for name,sha in json.loads((HERE/'manifest.json').read_text()).items():assert hashlib.sha256((HERE.parent/name).read_bytes()).hexdigest()==sha
    prior=json.loads((HERE.parent/'joint_strip_continuity/pilot.json').read_text());forms=prior['proof']['forms']
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth');sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==prior['checkpoint_sha256']
    sd=torch.load(path,map_location='cpu',weights_only=True);wi=tuple(F(float(v)) for v in sd['rnn.weight_ih_l0'][:,0]);wh=tuple(tuple(F(float(v)) for v in r) for r in sd['rnn.weight_hh_l0']);wo=tuple(tuple(F(float(v)) for v in r) for r in sd['linear.weight']);assert wi[0]<0<wi[1]
    unknown=[];records=[];zero_state=[]
    for i,form in enumerate(forms):
        A,B,C,_,E=map(F,form['row'])
        if not C:zero_state.append(i);continue
        if not A and not B:unknown.append(i);continue
        boundary=(C,A,B,E)
        for side in (0,1):
            original=[(F(1),F(0),F(0),F(0))]+faces.occurrence_constraints(form,forms,wi,wh,wo,0,side)
            reduced=[faces.eliminate(boundary,r) for r in original];p=project(reduced);w=None
            if p['feasible']:
                lo,hi=p['lower'],p['upper'];assert lo is not None and lo>=0
                s=(lo+hi)/2 if hi is not None else lo+1;t=witness(reduced,s)
                if abs(B)>=abs(A):x=t;y=-(C*s+A*x+E)/B
                else:y=t;x=-(C*s+B*y+E)/A
                assert C*s+A*x+B*y+E==0 and all(sum(a*b for a,b in zip(r,(s,x,y,1)))>=0 for r in original)
                w=(s,x,y)
            records.append(dict(source_index=i,kind=form['kind'],rank_side='01' if side==0 else '10',boundary=boundary,original_constraints=original,substituted_constraints=reduced,projection=p,witness=w))
    intervals=[(r['projection']['lower'],r['projection']['upper']) for r in records if r['projection']['feasible']];joined=union(intervals);gaps=None if unknown else complement(intervals)
    out=dict(checkpoint_sha256=sha,axis=0,target=F(1),prefix_condition='x0<0; h1=(-abs(Wi0)*x0,0); s>=0 is closed overapproximation',rank_condition='exactly one of x,y above1',input_limits=None,source_forms=len(forms),zero_state_forms=zero_state,unknown_forms=unknown,records=records,projected_face_count=len(records),feasible_faces=len(intervals),union=joined,sufficient_open_invariance_intervals=gaps,prefix_ratio_intervals=None if gaps is None else [[None if hi is None else hi/wi[0],lo/wi[0]] for lo,hi in gaps],cpu_seconds=time.process_time()-start,trained_forwards=0,checkpoint_loads=1,scope='Sufficient no-moving-boundary sectors only; covered intervals are inconclusive, constants/mass/integration uncomputed.')
    (HERE/'pilot.json').write_text(json.dumps(serial(out),indent=2)+'\n');print(json.dumps(serial({k:out[k] for k in ('projected_face_count','feasible_faces','unknown_forms','union','sufficient_open_invariance_intervals','prefix_ratio_intervals','cpu_seconds')})))
if __name__=='__main__':main()
