"""24 exact algebraic pairs and4 shape guards; no trained model or mass."""
from pathlib import Path
from fractions import Fraction as F
import itertools,json,hashlib,time
from canonical import canonical,margins,future_forms,direct,from_key,label,pack
H=Path(__file__).resolve().parent

def net(B,V,L=None):return dict(B=B,V=V,L=L)
def fixtures():
    e=[1,0];f=[0,1];base=net([e,f],[[1,2],[-1,3]]);cases=[]
    def add(name,a,b,exact=True,projective=None,labels=None):cases.append(dict(id=name,a=a,b=b,exact=exact,projective=exact if projective is None else projective,labels=labels))
    add('identity',base,base)
    add('hidden_permutation',base,net([f,e],[[2,1],[3,-1]]))
    add('duplicate_combination',base,net([e,e,f],[[F(1,3),F(2,3),2],[2,-3,3]]))
    add('positive_rescale',base,net([[2,0],[0,3]],[[F(1,2),F(2,3)],[-F(1,2),1]]))
    add('negative_orientation',net([[-1,0]],[[1]]),net([e],[[1]],[[-1,0]]))
    add('relu_difference_linear',net([e,[-1,0]],[[1,-1]]),net([], [[]],[[1,0]]))
    add('duplicate_cancellation',net([e,e],[[2,-2]]),net([], [[]],[[0,0]]))
    add('zero_form',net([e,[0,0]],[[1,99]]),net([e],[[1]]))
    add('zero_coefficient',net([e,f],[[1,0]]),net([e],[[1]]))
    add('common_output_shift',net([e,f],margins([[1,2],[3,4],[0,5]])),net([e,f],margins([[8,-3],[10,-1],[7,0]])))
    add('identical_readouts',net([e,f],margins([[3,-2],[3,-2],[3,-2]])),net([], [[],[]],[[0,0],[0,0]]),labels=True)
    add('positive_uniform_margin_scale',base,net([e,f],[[2,4],[-2,6]]),False,True,True)
    add('negative_margin_scale',base,net([e,f],[[-1,-2],[1,-3]]),False,False,False)
    add('different_hyperplane',net([e],[[1]]),net([[1,1]],[[1]]),False,False,False)
    add('linear_perturbation',net([e],[[1]]),net([e],[[1]],[[0,1]]),False,False,False)
    absforms=[e,[-1,0],f,[0,-1]]
    add('same_labels_not_same_margin',net(absforms,[[1,1,1,1]]),net(absforms,[[1,1,2,2]]),False,False,True)
    add('all_ties_choose_first',net([e],margins([[2],[2],[2]])),net([f],margins([[-3],[-3],[-3]])),labels=True)
    Wi=[1,-2];Wh=[[1,2],[0,1]];M=[[1,0],[0,1]];W=[[0,0],[1,-1]]
    add('future_bridge_explicit',net(future_forms(Wi,Wh,M),margins(W)),net([[1,2,1],[0,1,-2]],[[1,-1]]))
    add('zero_recurrence_different_states',net(future_forms(Wi,[[0,0],[0,0]],M),margins(W)),net(future_forms(Wi,[[0,0],[0,0]],[[7,8],[9,10]]),margins(W)))
    add('invisible_readout_different_states',net(future_forms(Wi,Wh,M),margins([[1,2],[1,2]])),net(future_forms(Wi,Wh,[[7,8],[9,10]]),margins([[1,2],[1,2]])))
    add('recurrence_null_state_difference',net(future_forms(Wi,[[1,0],[0,0]],M),margins(W)),net(future_forms(Wi,[[1,0],[0,0]],[[1,0],[7,8]]),margins(W)))
    add('opposed_outputs_same_hinge',net([[1,1],[1,1]],[[1,-1]]),net([], [[]],[[0,0]]))
    add('one_dimensional_orientation',net([[-2]],[[3]]),net([[1]],[[6]],[[-6]]))
    add('exact_extreme_positive_scale',net([[2**100,0]],[[F(1,2**100)]]),net([e],[[1]]))
    assert len(cases)==24;return cases

def dimension(n):return len(n['B'][0]) if n['B'] else len(n['L'][0])
def main():
    out=H/'controls.json';progress=H/'progress.jsonl';assert not out.exists() and not progress.exists();p=json.loads((H/'protocol.json').read_text())
    for f,h in p['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    def record(r):
        with progress.open('a') as f:f.write(json.dumps(pack(r))+'\n')
    rows=[];start=time.process_time();calls=0
    for c in fixtures():
        record(dict(stage='attempt',id=c['id']));a,b=c['a'],c['b'];d=dimension(a);assert d==dimension(b)
        ca=canonical(a['B'],a['V'],d,a['L']);cb=canonical(b['B'],b['V'],d,b['L']);same=ca['key']==cb['key'];projective=ca['projective']==cb['projective'];witnesses=[];same_labels=True;function_checks=True
        for x in itertools.product([-1,0,1],repeat=d):
            va=direct(a['B'],a['V'],x,a['L']);vb=direct(b['B'],b['V'],x,b['L']);calls+=2
            function_checks &= va==from_key(ca['key'],x) and vb==from_key(cb['key'],x)
            if va!=vb:witnesses.append(dict(x=x,a=va,b=vb))
            same_labels &= label(va)==label(vb)
        passed=same==c['exact'] and projective==c['projective'] and function_checks and (c['labels'] is None or same_labels==c['labels']) and (same or bool(witnesses))
        row=dict(id=c['id'],expected=dict(exact=c['exact'],projective=c['projective'],labels=c['labels']),a=a,b=b,canonical_a=ca,canonical_b=cb,exact_equal=same,projective_equal=projective,fixed_grid_label_equal=same_labels,function_checks=function_checks,difference_witness=witnesses[0] if witnesses else None,passed=passed);rows.append(row);record(dict(stage='result',row=row))
    guards=[]
    for name,args in [('bad_form_width',([[1,2]],[[1]],1,None)),('bad_coeff_width',([[1]],[[1,2]],1,None)),('no_outputs',([[1]],[],1,None)),('bad_linear_width',([[1]],[[1]],1,[[0,0]]))]:
        record(dict(stage='guard_attempt',id=name));rejected=False
        try:canonical(*args)
        except AssertionError:rejected=True
        row=dict(id=name,passed=rejected);guards.append(row);record(dict(stage='guard_result',row=row))
    result=dict(complete=True,cases=28,rows=rows,guards=guards,all_pass=all(r['passed'] for r in rows+guards),synthetic_direct_evaluations=calls,new_trained_geometry_mass_MC_calls=0,cpu_seconds=time.process_time()-start,scope=p['scope'])
    out.write_text(json.dumps(pack(result),indent=2)+'\n');print(json.dumps(dict(all_pass=result['all_pass'],cpu_seconds=result['cpu_seconds'],synthetic_direct_evaluations=calls)));assert result['all_pass'],'durable control failure'
if __name__=='__main__':
    try:main()
    except BaseException as e:
        with (H/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(type=type(e).__name__,message=str(e)))+'\n')
        raise
