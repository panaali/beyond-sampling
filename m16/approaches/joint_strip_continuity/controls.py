"""Exact fixtures before any trained application."""
from pathlib import Path
from fractions import Fraction as F
import importlib.util,json,time
from strips import certify,count_forms,serial,key
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('previous',HERE/'previous_certificate.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
start=time.process_time();wi=(F(1),F(1));zero=((F(0),F(0)),(F(0),F(0)));identity=((F(1),F(0)),(F(0),F(1)));diag=((F(1),F(0)),(F(0),F(2)));wo=((F(0),F(0)),(F(0),F(0)),(F(1),F(-1)),(F(0),F(0)))
a=certify((wi,zero,wo),2);assert a['sufficient'] and a['K_unique']==0
b=certify((wi,diag,((F(1),F(-1)),)*4),2);assert b['sufficient'] and b['K_unique']==0
c=certify((wi,diag,wo),2);p=old.certify((wi,diag,wo),2);assert c['sufficient'] and not p['sufficient'] and c['K_unique']>0
u=certify((wi,identity,wo),2);assert not u['sufficient'] and len(u['unknown'])>0
one=count_forms([dict(row=(F(0),F(1),F(-1),F(0),F(0)))]);assert one['K_unique']==1
base=(F(1),F(0),F(1),F(0),F(0));rows=[base,tuple(2*x for x in base),tuple(-3*x for x in base),(F(1),F(0),F(1),F(0),F(1))];d=count_forms([dict(row=r) for r in rows]);assert d['K_literal']==4 and d['K_unique']==2 and key(rows[0])==key(rows[2])
# Noncommuting dynamics independently evaluated as scalar forced-gate steps.
wi2=(F(2,3),F(-4,5));wh2=((F(1,2),F(3,4)),(F(-2,5),F(5,6)));wo2=((F(1,3),F(2,7)),(F(-3,5),F(4,9)),(F(5,11),F(-6,13)),(F(7,15),F(8,17)))
forms=certify((wi2,wh2,wo2),2)['forms'];coefficient_controls=0
for form in forms:
    for k in range(5):
        values=[F(int(j==k)) for j in range(5)];x,y,h0,h1,target=values;hidden=[h0,h1]
        for scalar,gate in ((x,form['first']),(target,form['middle'])):
            hidden=[(wi2[i]*scalar+wh2[i][0]*hidden[0]+wh2[i][1]*hidden[1])*int(gate[i]) for i in range(2)]
        z=[wi2[i]*y+wh2[i][0]*hidden[0]+wh2[i][1]*hidden[1] for i in range(2)]
        expected=z[form['neuron']] if form['kind']=='final_gate' else sum((wo2[2][i]-wo2[form['competitor']][i])*int(form['final'][i])*z[i] for i in range(2))
        assert expected==form['row'][k];coefficient_controls+=1
out=dict(noncommuting_exact_coefficient_controls=coefficient_controls,memoryless={k:a[k] for k in ('sufficient','K_unique','emitted')},constant_output={k:b[k] for k in ('sufficient','K_unique','emitted')},joint_rescue={k:c[k] for k in ('sufficient','K_literal','K_unique','emitted','unique_known')},previous_rescue_unknown=p['unknown'],true_unknown_count=len(u['unknown']),true_unknown_example=u['unknown'][0],one_dimensional_threshold_coefficient=one['K_unique'],dedup_literal=d['K_literal'],dedup_unique=d['K_unique'],cpu_seconds=time.process_time()-start)
(HERE/'controls.json').write_text(json.dumps(serial(out),indent=2)+'\n');print(json.dumps(serial(out)))
