"""Saved-only exact face inclusion and independent high-precision constant checks."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,sys,time
import mpmath as mp
sys.set_int_max_str_digits(0)
mp.mp.dps=180
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/prefix_wedge_remainder')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def m(q):
    q=F(q);return mp.mpf(q.numerator)/q.denominator
def interval(d):return F(d['lower']),F(d['upper'])
def contains(d,x):
    lo,hi=interval(d);assert lo<=hi and m(lo)<=x<=m(hi)
def event(d):
    k=F(d['k']);L=1/(8*mp.sqrt(2)*mp.pi*m(k))+mp.atan(1/mp.sqrt(2))/(4*mp.sqrt(2)*mp.pi**2*m(k))
    rem=1/(16*mp.pi*m(k)**3)
    contains(d['leading'],L);contains(d['cubic_remainder'],rem)
    lo,hi=interval(d['event_probability']);assert m(lo)<=max(0,L-rem) and m(hi)>=L
    contains(d['rank_factor_wedge_upper'],3*mp.atan(1/m(k))/(16*mp.pi))
    return k,lo,hi
start=time.process_time();c=read(P/'controls.json');a=read(P/'application.json')
for n,h in c['source_sha256'].items():assert sha(P/n)==h
assert sha(P/'application.py')==a['application_source_sha256']
gate=read(P/'control_gate.json');assert sha(P/'controls.json')==gate['controls_sha256'] and sha(P/'THEOREM.md')==gate['theorem_sha256']
contains(c['pi'],mp.pi)
for row in c['roots']:
    x=F(row['input']);lo,hi=interval(row['interval']);assert 0<=lo<=hi and lo*lo<=x<=hi*hi
    contains(row['interval'],mp.sqrt(m(x)))
for row in c['angles']:contains(row['interval'],mp.atan(m(row['input'])))
lo,hi=interval(c['addition_identity_zero_interval']);assert lo<=0<=hi
assert len(c['roots'])==6 and len(c['angles'])==9 and len(c['event_controls'])==4
for row,k0 in zip(c['event_controls'],(1,2,10,100)):
    k,lo,hi=event(row['bounds']);assert k==k0 and float(lo)<=row['scalar_integral']<=float(hi)
    if k==1:assert lo<=F(11,384)<=hi
assert a['complete'] and a['checkpoint_loads']==1 and a['network_forward_MC_calls']==0
for path,h in a['source_pins'].items():assert sha(Path(path))==h
prior=read(P.parent/'prefix_face_projection/pilot.json')
forms=read(P.parent/'joint_strip_continuity/pilot.json')['proof']['forms']
assert len(forms)==176 and not prior['unknown_forms'] and prior['axis']==0 and F(prior['target'])==1
moving={i for i,f in enumerate(forms) if F(f['row'][2])!=0}
zero=set(range(176))-moving;assert len(zero)==77 and zero==set(prior['zero_state_forms'])
assert {(r['source_index'],r['rank_side']) for r in prior['records']}=={(i,s) for i in moving for s in ('01','10')}
assert len(prior['records'])==198
wi=list(map(F,a['weights']['Wi']));wh=[list(map(F,row)) for row in a['weights']['Wh']]
assert wi[0]<0<wi[1] and wh[0][0]<0 and wh[1][0]<0
kap=-wh[1][0]/wi[1];k=kap*abs(wi[0]);assert F(a['kappa'])==kap and F(a['k'])==k
gate=(wh[1][0],wi[1],F(0),F(0));assert tuple(map(F,a['shared_gate_constraint']))==gate
matched=[]
for i,row in enumerate(prior['records']):
    if not row['projection']['feasible']:continue
    form=forms[row['source_index']];assert form['first']==[False,True] and row['rank_side']=='10'
    matches=[]
    for j,raw in enumerate(row['original_constraints']):
        r=tuple(map(F,raw));scale=r[0]/gate[0]
        if scale>0 and all(x==scale*y for x,y in zip(r,gate)):matches.append((j,scale,r))
    assert matches
    saved=next(e for e in a['feasible_face_evidence'] if e['record_index']==i)
    assert saved['source_index']==row['source_index'] and saved['kind']==row['kind'] and saved['rank_side']=='10'
    assert [(z['constraint_index'],F(z['scale']),tuple(map(F,z['row']))) for z in saved['matches']]==matches
    matched.append(i)
assert len(matched)==len(a['feasible_face_evidence'])==14
assert a['checked_complete_occurrence_rank_pairs']==198 and a['reused_infeasible_projection_certificates']==184
assert F(a['sector_probability'])==F(1,12) and F(a['rank10_sector_probability'])==F(1,24)
kb,lo,hi=event(a['bounds']);assert kb==k
assert float(lo)<=a['event_scalar_integral']<=float(hi)
# Check the k=1 event identity by exact polynomial integration, independent of references.
def primitive(t):return (t**4/F(4)-t**3/F(6)-t*t/F(8)+t/F(8))/2
assert primitive(F(1))-primitive(F(1,2))==F(11,384)
out=dict(complete=True,primitive_controls=20,complete_occurrence_rank_pairs=198,shared_faces=14,
         k=str(k),event_lower=str(lo),event_upper=str(hi),event_display=[float(lo),float(hi)],
         application_sha256=sha(P/'application.json'),controls_sha256=sha(P/'controls.json'),
         review_cpu=time.process_time()-start,new_model_quadrature_calls=0,
         scope='Exact saved shared-constraint/completeness audit, exact square-root brackets and independent 180-digit constant checks, backed by source interval proof. Exhaustive earlier face proof is reused. Upper bounds actual disagreement only on the stated sector; lower event bound is not an approximation-error lower bound. No full-model or production-cost claim.')
(HERE/'prefix_wedge_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('event_lower','event_upper')}))
