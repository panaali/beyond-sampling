"""Exact closed crossing faces; all rows use (s,x,y,constant) >= 0."""
from fractions import Fraction as F
from pathlib import Path
import importlib.util

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
G=module('axis_geometry',Path(__file__).parent.parent/'axis_strip_continuity/geometry.py')

def eliminate(boundary,row):
    C,A,B,E=boundary
    if not A and not B:raise ValueError('zero input normal')
    # Keep t=x when eliminating y, otherwise t=y.
    j=2 if abs(B)>=abs(A) else 1;k=3-j;n=boundary[j]
    return (row[0]-row[j]*C/n,row[k]-row[j]*boundary[k]/n,row[3]-row[j]*E/n)

def face(boundary,constraints,slo,shi,limit=F(6)):
    bounds=[(0,1,0,limit),(0,-1,0,limit),(0,0,1,limit),(0,0,-1,limit)]
    reduced=[eliminate(boundary,tuple(map(F,r))) for r in bounds+constraints]
    poly=G.rectangle(slo,shi,-limit,limit)
    for row in reduced:poly=G.clip(poly,row)
    return dict(feasible=bool(poly),vertices=poly,area=G.area(poly),constraints=reduced,eliminated='y' if abs(boundary[2])>=abs(boundary[1]) else 'x')

def damping(row,slo,shi):
    A,B,C,E=row
    if not A and not B:raise ValueError('zero input normal')
    low,high=sorted((C*slo+E,C*shi+E));offset=F(0) if low<=0<=high else min(abs(low),abs(high))
    q=offset*offset/(2*(A*A+B*B))
    return dict(min_absolute_offset=offset,q=q,factor=1/(1+q))

def occurrence_constraints(form,forms,wi,wh,wo,axis,side,u=F(1),label=2):
    first=tuple(form['first']);middle=tuple(form['middle']);word=(first,middle)
    rows=[(r[0],r[1],F(0),r[2]) for r in G.word_constraints(wi,wh,axis,first,middle,u)]
    # Need1: exactly one of x,y exceeds target u; closed sides overapproximate.
    rows += [(F(0),F(-1 if side==0 else 1),F(0),u if side==0 else -u),
             (F(0),F(0),F(1 if side==0 else -1),-u if side==0 else u)]
    if form['kind']=='margin':
        Z={f['neuron']:tuple(map(F,f['row'])) for f in forms if f['kind']=='final_gate' and (tuple(f['first']),tuple(f['middle']))==word}
        for i,on in enumerate(form['final']):
            A,B,C0,C1,E=Z[i];sign=1 if on else -1
            rows.append(tuple(sign*v for v in ((C0,C1)[axis],A,B,E*u)))
        for other in range(len(wo)):
            if other==label:continue
            delta=[(wo[label][i]-wo[other][i])*int(form['final'][i]) for i in range(2)]
            r=tuple(sum(delta[i]*Z[i][j] for i in range(2)) for j in range(5))
            rows.append((r[axis+2],r[0],r[1],r[4]*u))
    return rows
