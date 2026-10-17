"""Frozen T155 intervals; exact crossing-face and density ablations."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time
import torch
from faces import face,damping,occurrence_constraints,G
HERE=Path(__file__).parent

def main():
    start=time.process_time()
    for name,sha in json.loads((HERE/'manifest.json').read_text()).items():assert hashlib.sha256((HERE.parent/name).read_bytes()).hexdigest()==sha
    prior=json.loads((HERE.parent/'axis_strip_continuity/pilot.json').read_text());joint=json.loads((HERE.parent/'joint_strip_continuity/pilot.json').read_text());forms=joint['proof']['forms']
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth');sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==prior['checkpoint_sha256']
    sd=torch.load(path,map_location='cpu',weights_only=True);wi=tuple(F(float(v)) for v in sd['rnn.weight_ih_l0'][:,0]);wh=tuple(tuple(F(float(v)) for v in row) for row in sd['rnn.weight_hh_l0']);wo=tuple(tuple(F(float(v)) for v in row) for row in sd['linear.weight'])
    results=[];problem_count=0
    for old in prior['rows']:
        axis=old['axis'];slo,shi=map(F,old['hidden_interval']);records=[]
        for record in old['feasible_bound']['unique_records']:
            row=tuple(map(F,record['row']));A,B,C,E=row;occurrences=[]
            for idx in record['source_indices']:
                form=forms[idx];original=tuple(map(F,form['row']));r=(original[0],original[1],original[axis+2],original[4]);assert damping(r,slo,shi)['factor']==damping(row,slo,shi)['factor']
                for side in (0,1):
                    constraints=occurrence_constraints(form,forms,wi,wh,wo,axis,side)
                    result=face((r[2],r[0],r[1],r[3]),constraints,slo,shi);problem_count+=1
                    occurrences.append(dict(source_index=idx,rank_side='01' if side==0 else '10',boundary=(r[2],r[0],r[1],r[3]),original_constraints=constraints,**result))
            keep=any(r['feasible'] for r in occurrences);coefficient=F(record['coefficient']);d=damping(row,slo,shi)
            records.append(dict(row=row,coefficient=coefficient,density=d,keep=keep,occurrences=occurrences))
        Kword=F(old['feasible_bound']['K_unique']);Kface=sum((r['coefficient'] for r in records if r['keep']),F(0));Kdamp=sum((r['coefficient']*r['density']['factor'] for r in records if r['keep']),F(0));Konly=sum((r['coefficient']*r['density']['factor'] for r in records),F(0));scale=abs(wi[axis]);mass=joint['display']['joint_rank_masses'][1]
        assert Kdamp<=Kface<=Kword and Kdamp<=Konly<=Kword
        results.append(dict(axis=axis,hidden_interval=[slo,shi],prefix_interval=old['prefix_interval'],records=records,unique_before=len(records),unique_after=sum(r['keep'] for r in records),feasible_occurrences=sum(o['feasible'] for r in records for o in r['occurrences']),degenerate_feasible_occurrences=sum(o['feasible'] and not o['area'] for r in records for o in r['occurrences']),K_word=Kword,K_face=Kface,K_density_only=Konly,K_face_density=Kdamp,elementary_prefix=F(81,2)*scale*Kdamp,display=dict(K_word=float(Kword),K_face=float(Kface),K_density_only=float(Konly),K_face_density=float(Kdamp),elementary_prefix=float(F(81,2)*scale*Kdamp),CDF_prefix=float(scale*Kdamp)/mass,face_gain=float(Kword/Kface) if Kface else None,density_gain=float(Kface/Kdamp) if Kdamp else None)))
    out=dict(checkpoint_sha256=sha,rows=results,face_problems=problem_count,cpu_seconds=time.process_time()-start,model_forwards=0,checkpoint_loads=1,scope='Local truncated pair-function prefix continuity; no pair values, old interpolation repair or population/accuracy claim.')
    (HERE/'pilot.json').write_text(json.dumps(G.serial(out),indent=2)+'\n');print(json.dumps(dict(rows=[{k:r[k] for k in ('axis','unique_before','unique_after','feasible_occurrences','degenerate_feasible_occurrences','display')} for r in results],face_problems=problem_count,cpu_seconds=out['cpu_seconds'])))
if __name__=='__main__':main()
