from pathlib import Path
import itertools,json,time
from fractions import Fraction as F
from geometry import Geometry

def main():
    start=time.process_time();rows=[]
    cases=[('empty',(1,1),((0,-1),(1,-1)),(),False),
           ('mixed_zero_last',(-1,1),((0,0),(0,0)),(3,0),True),
           ('mixed_nonzero_last',(-1,1),((0,0),(0,0)),(0,1),False),
           ('zero_network',(0,0),((0,0),(0,0)),(1,-3,2),True),
           ('recession',(1,1),((1,0),(0,2)),(-100,-100,-100,-100),False),
           ('negative_pair',(1,1),((0,-1),(1,-1)),(-1,-1),True),
           ('negative_pair_after_positive',(1,1),((0,-1),(1,-1)),(3,-1,-1),True)]
    for name,wi,wh,word,expected in cases:
        g=Geometry(wi,wh,seconds=15);r=g.check(word);r['name']=name;r['expected']=expected
        if r['global_reset']!=expected:raise AssertionError((name,r))
        radius=F(r['robust_radius']);corners=[]
        if radius:
            for signs in itertools.product((-1,1),repeat=len(word)):
                newword=[F(x)+s*radius/2 for x,s in zip(word,signs)]
                result=Geometry(wi,wh,seconds=15).check(newword)
                if not result['global_reset']:raise AssertionError('robust corner fails')
                corners.append({'word':result['word'],'global_reset':True,'cpu_seconds':result['cpu_seconds'],'stats':result['stats']})
        r['half_radius_corners']=corners;rows.append(r)
    result={'cases':rows,'cpu_seconds':time.process_time()-start,'sampled_inputs':0,'method':'exact Fraction projective polygon partition'}
    Path(__file__).with_name('controls_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cpu_seconds':result['cpu_seconds'],'cases':[{'name':r['name'],'reset':r['global_reset'],'radius':r['robust_radius'],'cells':r['final_cells'],'recession_vertices':r['recession_vertices']} for r in rows]},indent=2))
if __name__=='__main__':main()
