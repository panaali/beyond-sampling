"""Two frozen axis neighborhoods, exact feasible-word strip sums."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time
import torch
from geometry import feasible_words,serial
HERE=Path(__file__).parent;ROOT=HERE.parents[2]

def bound(forms,axis,allowed=None):
    total=F(0);unique={};unknown=[];nonzero=0;considered=0
    for i,form in enumerate(forms):
        word=(tuple(form['first']),tuple(form['middle']))
        if allowed is not None and word not in allowed:continue
        considered+=1;original=list(map(F,form['row']));row=(original[0],original[1],original[axis+2],original[4]);A,B,C,E=row
        if not C:continue
        nonzero+=1;normal=max(abs(A),abs(B))
        if not normal:unknown.append(i);continue
        coefficient=abs(C)/normal;total+=coefficient;pivot=next(v for v in row if v);key=tuple(v/pivot for v in row)
        if key in unique:
            assert coefficient==unique[key]['coefficient'];unique[key]['source_indices'].append(i)
        else:unique[key]=dict(row=row,coefficient=coefficient,source_indices=[i])
    return dict(sufficient=not unknown,K_literal=None if unknown else total,K_unique=None if unknown else sum((r['coefficient'] for r in unique.values()),F(0)),considered_forms=considered,nonzero_axis_forms=nonzero,unique_count=len(unique),unknown=unknown,unique_records=list(unique.values()))

def main():
    start=time.process_time()
    for name,sha in json.loads((HERE/'manifest.json').read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
    previous=json.loads((HERE.parent/'joint_strip_continuity/pilot.json').read_text());path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth');sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==previous['checkpoint_sha256'];sd=torch.load(path,map_location='cpu',weights_only=True);wi=tuple(F(float(v)) for v in sd['rnn.weight_ih_l0'][:,0]);wh=tuple(tuple(F(float(v)) for v in row) for row in sd['rnn.weight_hh_l0']);assert wi[0]<0<wi[1]
    forms=previous['proof']['forms'];rank_mass=previous['display']['joint_rank_masses'][1];fullK=F(previous['proof']['K_unique']);rows=[]
    for axis,lo,hi in ((0,F(-3,8),F(-1,8)),(1,F(1,8),F(3,8))):
        slo,shi=sorted((wi[axis]*lo,wi[axis]*hi));words=feasible_words(wi,wh,axis,slo,shi);allowed={(tuple(r['first']),tuple(r['middle'])) for r in words if r['feasible']};allwords=bound(forms,axis);pruned=bound(forms,axis,allowed);scale=abs(wi[axis]);display=None;certified=None
        if allwords['sufficient'] and pruned['sufficient']:
            A=allwords['K_unique'];K=pruned['K_unique'];certified=F(81,2)*K*scale
            display=dict(full_hidden_K=float(fullK),all_word_axis_K=float(A),feasible_axis_K=float(K),prefix_scale=float(scale),need1_prefix_CDF_bound=float(K*scale)/rank_mass,elementary_prefix_bound=float(certified),axis_bound_gain=float(fullK/A) if A else None,feasibility_gain=float(A/K) if K else None,CDF_display_not_outward=True)
        rows.append(dict(axis=axis,prefix_interval=[lo,hi],hidden_interval=[slo,shi],target=F(1),x_limits=[F(-6),F(6)],feasible_word_count=len(allowed),degenerate_feasible_words=sum(r['feasible'] and r['area']==0 for r in words),words=words,all_word_bound=allwords,feasible_bound=pruned,elementary_prefix_coefficient=certified,display=display))
    out=dict(checkpoint_sha256=sha,rows=rows,trained_forward_evaluations=0,word_feasibility_problems=32,cpu_seconds=time.process_time()-start,scope='Local axis/prefix-coordinate continuity only; no new pair values, interpolation error, global accuracy or estimator cost.')
    (HERE/'pilot.json').write_text(json.dumps(serial(out),indent=2)+'\n');print(json.dumps(serial(dict(rows=[{k:r[k] for k in ('axis','feasible_word_count','degenerate_feasible_words','display')} for r in rows],cpu_seconds=out['cpu_seconds']))))
if __name__=='__main__':main()
