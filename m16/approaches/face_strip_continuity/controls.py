from fractions import Fraction as F
from pathlib import Path
import json,time
from faces import face,damping,eliminate,occurrence_constraints,G,module

def main():
    start=time.process_time();cases=[]
    # y=0: full face, empty, line, point; and x=0 alternate elimination.
    for name,boundary,rows,want,dimension in [
        ('full_y',(0,0,1,0),[],True,2),
        ('empty',(0,0,1,0),[(0,0,1,-1)],False,None),
        ('line',(0,0,1,0),[(1,0,0,0),(-1,0,0,0)],True,1),
        ('point',(0,0,1,0),[(1,0,0,0),(-1,0,0,0),(0,1,0,0),(0,-1,0,0)],True,0),
        ('full_x',(0,1,0,0),[],True,2),
        ('rank01',(0,1,1,-2),[(0,-1,0,1),(0,0,1,-1)],True,2),
        ('rank10',(0,1,1,-2),[(0,1,0,-1),(0,0,-1,1)],True,2),
        ('wrong_face',(0,0,1,0),[(0,0,1,-1)],False,None)]:
        b=tuple(map(F,boundary));got=face(b,rows,F(0),F(1));assert got['feasible']==want
        if want:assert (2 if got['area'] else (1 if len(set(got['vertices']))>1 else 0))==dimension
        cases.append(dict(name=name,boundary=b,result=got))
    for b in [(F(2),F(3),F(1),F(-5)),(F(2),F(1),F(3),F(-5))]:
        row=(F(7),F(11),F(13),F(17));r=eliminate(b,row)
        for s,t in [(F(0),F(0)),(F(1,3),F(-2,7))]:
            if abs(b[2])>=abs(b[1]):x=t;y=-(b[0]*s+b[1]*x+b[3])/b[2]
            else:y=t;x=-(b[0]*s+b[2]*y+b[3])/b[1]
            assert sum(a*v for a,v in zip(row,(s,x,y,1)))==r[0]*s+r[1]*t+r[2]
    ds=[]
    for row,want in [((1,0,1,-F(1,2)),F(0)),((1,0,1,1),F(1,2)),((1,0,-1,-1),F(1,2)),((1,0,0,2),F(2))]:
        row=tuple(map(F,row));d=damping(row,F(0),F(1));assert d['q']==want
        assert d=={**damping(tuple(-3*v for v in row),F(0),F(1)),'min_absolute_offset':d['min_absolute_offset']}
        ds.append(dict(row=row,result=d))
    S=module('control_strips',Path(__file__).parent.parent/'joint_strip_continuity/strips.py')
    wi=(F(1),F(1));wh=((F(1),F(0)),(F(0),F(1)));wo=((F(2),F(2)),(F(1),F(0)),(F(1),F(1)),(F(0),F(0)))
    forms=S.certify((wi,wh,wo),2)['forms'];correctness=[]
    for f in forms:
        if f['kind']!='margin' or not all(f['first']) or not all(f['middle']) or not all(f['final']) or f['competitor']!=1:continue
        r=f['row'];boundary=(r[2],r[0],r[1],r[4])
        for side in (0,1):
            rows=occurrence_constraints(f,forms,wi,wh,wo,0,side)
            # Chosen margin z1=0 can occur with z0>0; higher competing row excludes correctness.
            full=face(boundary,rows,F(1),F(2));broad=face(boundary,rows[:8],F(1),F(2))
            assert not full['feasible']
            correctness.append(dict(side=side,full=full,broad=broad))
    assert len(correctness)==2 and [r['broad']['feasible'] for r in correctness]==[False,True]
    # Same geometric strip may have multiple faces: retain if any closed face survives.
    union=[face((F(0),F(0),F(1),F(0)),c,F(0),F(1)) for c in [[(0,0,1,-1)],[]]]
    assert any(f['feasible'] for f in union) and not all(f['feasible'] for f in union)
    out=dict(cases=cases,density=ds,correctness=correctness,elimination_identity_checks=4,any_occurrence_control=union,cpu_seconds=time.process_time()-start)
    Path(__file__).with_suffix('.json').write_text(json.dumps(G.serial(out),indent=2)+'\n');print(out['cpu_seconds'])
if __name__=='__main__':main()
