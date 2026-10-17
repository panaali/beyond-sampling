from pathlib import Path
from fractions import Fraction as F
import hashlib,itertools,json,time
from geometry import Geometry

def main():
    start=time.process_time()
    import torch
    path=Path('/workspace/clean-neurons/work/checkpoints/H2_T10_seed0.pth')
    state=torch.load(path,map_location='cpu',weights_only=True)
    wi=state['rnn.weight_ih_l0'].reshape(-1).tolist();wh=state['rnn.weight_hh_l0'].tolist()
    a,b,c,d=F(wh[0][0]),-F(wh[0][1]),F(wh[1][0]),-F(wh[1][1])
    theorem={'opposite_input_signs':wi[0]<0<wi[1],'positive_parameters':min(a,b,c,d)>0,'bc_gt_ad':b*c>a*d,'a_squared_lt_bc':a*a<b*c,'a_gt_d':a>d,'five_word_condition':1>d+F(1,20)/d}
    if not all(theorem.values()):raise AssertionError(theorem)
    cases=[('zero2',(0,0),False),('zero3',(0,0,0),True),('known_positive5',(1,1,1,1,F(1,20)),True)]
    for magnitudes in ((1,1,1,1),(1,1,3,1)):
        for signs in itertools.product((-1,1),repeat=4):
            cases.append(('four_'+''.join('p' if s>0 else 'm' for s in signs)+'_'+str(magnitudes),tuple(s*x for s,x in zip(signs,magnitudes)),None))
    result={'wi':wi,'wh':wh,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'theorem_conditions':theorem,'setup_cpu_seconds':time.process_time()-start,'cases':[],'sampled_inputs':0}
    out=Path(__file__).with_name('pilot_results.json')
    for name,word,expected in cases:
        t=time.process_time()
        if t-start>100:
            result['budget_stop_before']=name;break
        try:
            row=Geometry(wi,wh,seconds=100-(t-start)).check(word);row['name']=name
            if expected is not None and row['global_reset']!=expected:raise AssertionError((name,'known theorem fails'))
            radius=F(row['robust_radius']);corners=[]
            if radius:
                for signs in itertools.product((-1,1),repeat=len(word)):
                    newword=[F(x)+s*radius/2 for x,s in zip(word,signs)]
                    corner=Geometry(wi,wh,seconds=max(.01,100-(time.process_time()-start))).check(newword)
                    corners.append({'word':corner['word'],'global_reset':corner['global_reset'],'cpu_seconds':corner['cpu_seconds'],'stats':corner['stats']})
                    if not corner['global_reset']:raise AssertionError('certified corner fails')
            row['half_radius_corners']=corners;row['case_total_cpu_seconds']=time.process_time()-t
        except Exception as e:
            row={'name':name,'word':[str(x) for x in word],'failure':repr(e),'case_total_cpu_seconds':time.process_time()-t}
        result['cases'].append(row);result['cpu_seconds']=time.process_time()-start
        out.write_text(json.dumps(result,indent=2)+'\n')
        if 'failure'in row:break
    print(json.dumps({'cpu_seconds':result['cpu_seconds'],'cases':len(result['cases']),'four_word_resets':[{'name':r['name'],'radius':r['robust_radius'],'cells':r['final_cells']} for r in result['cases'] if r['name'].startswith('four') and r.get('global_reset')],'known':[{'name':r['name'],'reset':r.get('global_reset'),'radius':r.get('robust_radius'),'failure':r.get('failure')} for r in result['cases'][:3]],'failures':[r for r in result['cases'] if 'failure'in r]},indent=2))
if __name__=='__main__':main()
