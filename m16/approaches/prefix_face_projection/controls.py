from pathlib import Path
from fractions import Fraction as F
import json,time
import importlib.util
from project import project,witness,union,complement

def serial(v):
    if isinstance(v,F):return str(v)
    if isinstance(v,dict):return {k:serial(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [serial(x) for x in v]
    return v

def main():
    start=time.process_time();cases=[]
    definitions=[('all',[],True,None,None),('empty',[(0,1,0),(0,-1,-1)],False,None,None),('positive',[(1,1,0),(0,-1,0)],True,F(0),None),('negative',[(-1,1,0),(0,-1,0)],True,None,F(0)),('bounded',[(1,1,0),(0,-1,0),(-1,0,2)],True,F(0),F(2)),('point',[(1,1,-1),(-1,-1,1),(0,1,0),(0,-1,0)],True,F(1),F(1)),('constant_true',[(0,0,1)],True,None,None),('constant_false',[(0,0,-1)],False,None,None),('one_sided_t',[(1,1,0),(-1,1,0)],True,None,None)]
    for name,raw,ok,lo,hi in definitions:
        rows=[tuple(map(F,r)) for r in raw];result=project(rows);assert result['feasible']==ok
        if ok:
            assert (result['lower'],result['upper'])==(lo,hi)
            s=(lo+hi)/2 if lo is not None and hi is not None else lo if lo is not None else hi if hi is not None else F(0);t=witness(rows,s)
        else:s=t=None
        scaled=project([tuple((i+2)*x for x in row) for i,row in enumerate(rows)]);assert (scaled['feasible'],scaled['lower'],scaled['upper'])==(result['feasible'],result['lower'],result['upper'])
        # Every reduced inequality is an exact nonnegative original-row combination.
        for r in result['reduced']:
            total=tuple(sum(scale*rows[i][j] for i,scale in r['sources']) for j in range(3));assert all(scale>0 for _,scale in r['sources']);assert total==(r['row'][0],0,r['row'][1])
        cases.append(dict(name=name,rows=rows,result=result,witness=[s,t]))
    ints=[(F(1),F(2)),(F(2),F(3)),(F(5),F(5)),(F(7),None)]
    assert union(ints)==[[F(1),F(3)],[F(5),F(5)],[F(7),None]]
    assert complement(ints)==[[F(0),F(1)],[F(3),F(5)],[F(5),F(7)]]
    assert complement([])==[[F(0),None]] and complement([(F(0),None)])==[]
    spec=importlib.util.spec_from_file_location('face_controls',Path(__file__).parent.parent/'face_strip_continuity/faces.py');faces=importlib.util.module_from_spec(spec);spec.loader.exec_module(faces)
    substitution=[]
    for boundary,extra,expected in [((1,1,1,0),[],(True,F(0),None)),((1,1,1,0),[(0,1,0,0)],(False,F(0),F(-1))),((-1,1,1,0),[(0,1,0,0)],(True,F(1),None))]:
        boundary=tuple(map(F,boundary));rows=[(1,0,0,0),(0,-1,0,1),(0,0,1,-1)]+extra
        reduced=[faces.eliminate(boundary,tuple(map(F,r))) for r in rows];r=project(reduced)
        assert (r['feasible'],r['lower'],r['upper'])==expected
        substitution.append(dict(boundary=boundary,original_rows=rows,substituted=reduced,result=r))
    out=dict(cases=cases,substitution_controls=substitution,interval_union_controls=3,cpu_seconds=time.process_time()-start)
    Path(__file__).with_suffix('.json').write_text(json.dumps(serial(out),indent=2)+'\n');print(out['cpu_seconds'])
if __name__=='__main__':main()
