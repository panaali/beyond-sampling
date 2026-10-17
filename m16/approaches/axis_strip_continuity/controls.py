"""Exact primitive and synthetic chronology controls before trained data."""
from pathlib import Path
from fractions import Fraction as F
import json,time
from geometry import clip,rectangle,area,feasible_words,serial
start=time.process_time();box=rectangle(F(0),F(1),F(0),F(1));half=clip(box,(F(-1),F(0),F(1,2)));assert area(half)==F(1,2)
empty=clip(box,(F(1),F(0),F(-2)));assert not empty
line=clip(box,(F(1),F(0),F(-1)));assert len(line)==2 and area(line)==0 and all(p[0]==1 for p in line)
point=clip(line,(F(0),F(1),F(-1)));assert point==[(F(1),F(1))]
assert clip(box,(F(0),F(0),F(0)))==box and not clip(box,(F(0),F(0),F(-1)))
assert clip(half,(F(-2),F(0),F(1)))==half
wi=(F(1),F(1));wh=((F(1),F(0)),(F(0),F(1)));axes=[]
for axis in range(2):
    rows=feasible_words(wi,wh,axis,F(1),F(2));actual=[(r['first'],r['middle']) for r in rows if r['feasible']];expected=[(g,(True,True)) for g in ((False,False),((True,False) if axis==0 else (False,True)),(True,True))];assert set(actual)==set(expected);axes.append(dict(axis=axis,feasible=actual,total=len(rows)))
out=dict(half_area=area(half),empty_retained_as_empty=not empty,line_vertices=line,point_vertices=point,constant_true_false_and_duplicate_pass=True,synthetic_axes=axes,geometry_cases=7,synthetic_word_checks=32,cpu_seconds=time.process_time()-start)
Path(__file__).with_name('controls.json').write_text(json.dumps(serial(out),indent=2)+'\n');print(json.dumps(serial(out)))
