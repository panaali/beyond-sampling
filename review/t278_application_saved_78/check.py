"""T278 saved application contraction and fixed exact recurrence subset."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations,product
import gzip,hashlib,json,math,resource,time,traceback
from exact import decode,reconstruct,errors,difference,merged
P=Path(__file__).resolve().parent;START=time.process_time()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(pins):
 assert isinstance(pins,dict) and pins
 for p,h in pins.items():assert sha(p)==h,p
def save(r):
 r.update(cpu_seconds=time.process_time()-START,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 q=P/'results.tmp';q.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');q.replace(P/'results.json')
def cap():assert time.process_time()-START<35,'internal reviewer cap'
def finite(x):assert type(x) in (int,float) and math.isfinite(x)
def close(a,b,tol=2e-12):
 finite(a);finite(b);assert abs(a-b)<=tol*max(1.,abs(b)),(a,b)
def fsum(xs):return float(sum((F.from_float(float(x)) for x in xs),F()))
def timed_value(x):finite(x);assert x>=0

def main(r):
 scope=read(P/'scope.json');gate=read(P/'gate.json');ci=read(P/'completed_inputs.json');review=read(gate['source_review'])
 own={str(P/n) for n in ('check.py','exact.py','scope.json','README.md')}
 assert own<=set(review['pins']) and own|{str(P/'completed_inputs.json'),gate['source_review']}<=set(gate['pins'])
 assert gate['ready'] is True and gate['source_review_publication'] and gate['execution_publication']
 assert all(review[k] is True for k in ('complete','pass_all','execution_authorized'))
 for m in (scope['pins'],gate['pins'],ci['pins'],review['pins']):verify(m)
 a=Path(scope['author']);out=a/'application_output';p=read(a/'application_protocol.json');g=read(a/'application_gate.json');ar=read(p['source_review']);cr=read(p['controls_review'])
 required={str(a/n) for n in ('run_application.py','application_math.py','application_protocol.json','APPLICATION.md','application_launch.sh','application_gate.json','application_run1.log','intervals.py','gaussian.py')}|{str(out/n) for n in ('results.json','catalogue.json','points.jsonl','progress.jsonl','active_point.json')}|{p['source_review'],p['controls_review'],p['model_archive'],p['reference']}
 assert required<=set(ci['pins']) and ci['raw_publication']
 assert g['ready'] is True and all(ar[k] is True for k in ('complete','pass_all','execution_authorized')) and cr['complete'] is True and cr['pass_all'] is True and cr['rows']==18 and cr['events']==38
 for m in (p['pins'],g['pins'],ar['pins'],cr['input_pins']):verify(m)
 assert resource.getrlimit(resource.RLIMIT_CPU)==(39,40) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,)*2
 d=decode(read(out/'results.json'));assert d['pins']==g['pins'] and d['rng_calls']==0 and d['reference_recomputed'] is False
 complete=d['complete'] is True and d['pass_all'] is True
 if complete:assert d['points']==2718 and 'failure' not in d and len(d['summary'])==2
 else:
  assert d['complete'] is False and d['pass_all'] is False and d['points']==18 and d['summary']==[]
  assert 'predeclared resource projection exceeds75CPU' in d['failure'];assert d['resource_projection']['naive_full_projection']>75
 assert d['source_face_calls']==d['points'] and d['model_partition_calls']==2*d['points']
 finite(d['cpu_seconds']);assert 0<=d['cpu_seconds']<110 and type(d['maxrss_kib'])is int and d['maxrss_kib']>0
 rawmodel=read(p['reference']);assert rawmodel['complete'] and rawmodel['H']==rawmodel['T']==4 and rawmodel['exact_artifact_sha256']==p['pins'][p['model_archive']] and d['reference_interval']==rawmodel['accuracy']['probability_interval'];reflo,refhi=d['reference_interval'];assert 0<reflo<refhi<1 and [reflo.hex(),refhi.hex()]==rawmodel['accuracy']['probability_interval_hex']
 archive=json.loads(gzip.decompress(Path(p['model_archive']).read_bytes()));model=decode(d['model']['weights'])
 def fracpair(v):
  assert len(v)==2 and all(type(x)is str and str(int(x))==x for x in v) and int(v[1])>0;return F(*map(int,v))
 expected=[[fracpair(v) for v in archive['model'][0]],[[fracpair(v) for v in row] for row in archive['model'][1]],[[fracpair(v) for v in row] for row in archive['model'][2]]]
 assert model==expected and d['model']['checkpoint_sha256']==archive['checkpoint_sha256'];del archive
 assert len(model[0])==4 and len(model[1])==len(model[2])==4 and all(len(x)==4 for x in model[1]+model[2])
 controls=read(a/'controls_output/results.json');parity=[]
 for row in controls['rows']:
  if row['kind']=='gaussian':parity.append({m:row['value'][m].hex() for m in ('first','second','direct')})
 assert d['synthetic_formula_parity']==parity
 # Saved GL nodes/weights: finite moment identities, not fresh rule construction.
 assert set(d['rules'])=={'3','5'};rules={}
 for order in (3,5):
  rr=d['rules'][str(order)];nodes=list(map(float.fromhex,rr['nodes']));weights=list(map(float.fromhex,rr['weights']));assert len(nodes)==len(weights)==order and nodes==sorted(set(nodes)) and all(0<x<1 for x in nodes) and all(x>0 for x in weights)
  for degree in range(2*order):close(fsum(w*x**degree for w,x in zip(weights,nodes,strict=True)),1/(degree+1),2e-14)
  timed_value(rr['setup_cpu']);rules[order]=(nodes,weights)
 cat=read(out/'catalogue.json');assert sha(out/'catalogue.json')==d['catalogue_sha256']
 configs={'12':[(a,b,-1) for a,b in combinations(range(4),2)],'23':[(a,b,t) for a,b in combinations(range(4),2) for t in range(4) if t not in (a,b)]}
 catalogue=cat['catalogue'];assert len(catalogue)==2718
 lookup={}
 for i,row in enumerate(catalogue):
  assert set(row)=={'kind','config','unit_hex'} and row['kind'] in configs and type(row['config'])is int and 0<=row['config']<len(configs[row['kind']]) and len(row['unit_hex'])==3
  vals=[float.fromhex(x) for x in row['unit_hex']];assert all(0<x<1 and x.hex()==h for x,h in zip(vals,row['unit_hex'],strict=True))
  key=(row['kind'],row['config'],tuple(row['unit_hex']));assert key not in lookup;lookup[key]=i
 centres=[dict(kind=k,config=i,unit_hex=[(.5).hex()]*3) for k in ('23','12') for i in range(len(configs[k]))];assert catalogue[:18]==centres
 Z=d['face_setup']['Z'];assert len(Z)==2 and all(type(x)is float and math.isfinite(x) and x>0 for x in Z) and d['face_setup']['grid_points']==160001
 memberships={};assert set(cat['members'])=={'3','5'}
 for order in (3,5):
  nodes,weights=rules[order];expected_members=[]
  for kind,z in zip(('23','12'),Z,strict=True):
   for ci0 in range(len(configs[kind])):
    for ids in product(range(order),repeat=3):
     key=(kind,ci0,tuple(nodes[j].hex() for j in ids));idx=lookup[key];weight=z/len(configs[kind])*math.prod(weights[j] for j in ids);expected_members.append([idx,weight])
  assert cat['members'][str(order)]==expected_members and len(expected_members)==18*order**3;memberships[order]=expected_members
 assert {idx for rows in memberships.values() for idx,w in rows}==set(range(2718))
 selected={j*2717//63 for j in range(64)};assert cat['full_cell_record_ids']==d['full_cell_record_ids']==sorted(selected)
 rows=[];recurrences=0;assignments=0;maxwindow=0.;eventraw=[json.loads(x) for x in (out/'progress.jsonl').read_text().splitlines()];phases={};bytecount=0
 with (out/'points.jsonl').open() as stream:
  for i,line in enumerate(stream):
   cap();bytecount+=len(line.encode());row=decode(json.loads(line));item=catalogue[i];assert row['index']==i and all(row[k]==v for k,v in item.items())
   a0,b0,top=configs[item['kind']][item['config']];y=row['y'];assert len(y)==4 and all(type(x)is F for x in y) and y[a0]==y[b0]
   others=[y[j] for j in range(4) if j not in (a0,b0) and (item['kind']!='23' or j!=top)];low=max(others)
   width=min((y[top]-y[a0])/3,y[a0]-low) if item['kind']=='23' else (y[a0]-low)/3
   width=min(width,F.from_float(8.5/math.sqrt(2)));assert width==row['window_delta'] and width>0
   finite(row['source_tau_z']);maxwindow=max(maxwindow,abs(math.sqrt(2)*float(width)-row['source_tau_z']))
   assert len(row['layers'])==4 and all(type(x)is int and x>0 for x in row['layers']) and row['frozen_layers']==[1]*4
   direction=[F(int(j==a0)-int(j==b0)) for j in range(4)]
   if i in selected:
    cells,layers,frozen,fl,na=reconstruct(model,y,direction,width);assert row['cells']==cells and row['layers']==layers and row['frozen_cells']==frozen and row['frozen_layers']==fl;recurrences+=1;assignments+=na
   else:assert 'cells' not in row and 'frozen_cells' not in row
   assert set(row['cpu'])=={'face','actual_partition','frozen_partition'}
   for key,v in row['cpu'].items():timed_value(v);phases.setdefault('face_points' if key=='face' else key,[]).append(v)
   assert len(row['halves'])==2
   for h,(side,lo,hi,truth) in zip(row['halves'],[('negative',-width,F(0),b0 if item['kind']=='23' else a0),('positive',F(0),width,a0 if item['kind']=='23' else b0)],strict=True):
    assert h['side']==side and h['truth']==truth
    for key in ('actual_errors','frozen_errors'):
     intervals=h[key];assert intervals==merged(intervals) and all(lo<=l<u<=hi for l,u in intervals)
    if i in selected:assert h['actual_errors']==errors(cells,truth,lo,hi) and h['frozen_errors']==errors(frozen,truth,lo,hi)
    assert h['bands']==difference(h['actual_errors'],h['frozen_errors']) and all(lo<=l<u<=hi and s in (-1,1) for l,u,s in h['bands'])
    assert set(h['correction'])==set(h['terms'])=={'first','second','direct'}
    for key in ('window_mass','actual_mass','frozen_mass'):finite(h[key]);assert -1e-15<=h[key]<=.5+1e-15
    assert h['actual_mass']<=h['window_mass']+2e-14 and h['frozen_mass']<=h['window_mass']+2e-14
    for mode in ('first','second','direct'):
     assert len(h['terms'][mode])==len(h['bands']);assert all(type(x)is float and math.isfinite(x) for x in h['terms'][mode]);close(h['correction'][mode],fsum(h['terms'][mode]),2e-15)
    # First/second formulas only. Direct erf/CDF values stay provenance-qualified.
    for j,(l,u,sgn) in enumerate(h['bands']):
     c=l if l>=0 else u;delta=u-l if l>=0 else l-u;cf,df=float(c),float(delta);q=1 if df>0 else -1;density=math.exp(-cf*cf)/math.sqrt(math.pi)
     close(h['terms']['first'][j],sgn*q*density*df,2e-14);close(h['terms']['second'][j],sgn*q*density*df*(1-cf*df),2e-14)
    widths=[float(u-l) for l,u,sgn in h['bands']]
    close(h['bound1'],math.exp(-.5)/math.sqrt(2*math.pi)*fsum(x*x for x in widths),2e-14);close(h['bound2'],fsum(abs(x)**3 for x in widths)/(3*math.sqrt(math.pi)),2e-14)
    assert h['parity_error']==abs((h['actual_mass']-h['frozen_mass'])-h['correction']['direct'])<=3e-13
    assert set(h['cpu'])=={'actual_errors','frozen_errors','matching','actual_mass','frozen_mass','first','second','direct'}
    for k,v in h['cpu'].items():
     timed_value(v)
     if k not in ('actual_mass','frozen_mass'):phases.setdefault(k,[]).append(v)
   assert row['ledger'] and all(type(v)is int and v>=0 for v in row['ledger'].values()) and len(row['returned_bit_max'])==2 and all(type(v)is int and v>0 for v in row['returned_bit_max'])
   rows.append(row)
   if i%64==0:r.update(points_checked=len(rows),recurrences_checked=recurrences);save(r)
 assert len(rows)==d['points'] and bytecount==d['raw_bytes']<=128*2**20
 assert recurrences==len(selected&set(range(d['points'])))
 active=decode(read(out/'active_point.json'));assert all(active[k]==v for k,v in rows[-1].items() if k not in ('ledger','returned_bit_max'))
 assert set(d['cpu_phases'])==set(phases)
 for k,values in phases.items():close(d['cpu_phases'][k],fsum(values))
 projection=d['resource_projection'];assert projection['pilot_points']==18 and projection['limit']==75;timed_value(projection['recorded_cpu'])
 close(projection['naive_full_projection'],projection['recorded_cpu']+(projection['recorded_cpu']-d['model_and_reference_load_cpu']-d['face_setup']['cpu'])*150)
 if complete:assert projection['naive_full_projection']<=75 and sha(out/'points.jsonl')==d['points_sha256'] and d['max_normal_window_difference_from_source']==maxwindow
 expected_events=[dict(event='started'),dict(event='formula_parity_completed')]+[dict(event=e,index=i) for i in range(len(rows)) for e in ('point_started','point_completed')]+([dict(event='completed')] if complete else [])
 assert len(eventraw)==d['events']==len(expected_events)==(5439 if complete else 38);last=0.
 for raw,expected in zip(eventraw,expected_events,strict=True):
  stamp=raw.pop('cpu');finite(stamp);assert last<=stamp<=d['cpu_seconds'];last=stamp;assert raw==expected
 metrics_out=[]
 if complete:
  for saved,order in zip(d['summary'],(3,5),strict=True):
   assert saved['order']==order;members=memberships[order];unique={i for i,w in members};keys=['zero','first','second','direct','actual','bound1','bound2','window_mass'];strata={f'{kind}:{k}':{m:[] for m in keys} for kind in ('23','12') for k in range(4)}
   for i,w in members:
    row=rows[i]
    for h in row['halves']:
     ss=strata[f'{row["kind"]}:{h["truth"]}'];ss['zero'].append(w*h['frozen_mass']);ss['actual'].append(w*h['actual_mass']);ss['window_mass'].append(w*h['window_mass'])
     for mode in ('first','second','direct'):ss[mode].append(w*(h['frozen_mass']+h['correction'][mode]))
     for mode in ('bound1','bound2'):ss[mode].append(w*h[mode])
   assert set(saved['strata'])==set(strata)
   sums={k:{m:fsum(vals) for m,vals in v.items()} for k,v in strata.items()}
   for k,v in sums.items():
    assert set(saved['strata'][k])==set(keys)
    for m,x in v.items():close(saved['strata'][k][m],x,2e-14)
   assert set(saved['totals'])==set(keys)
   for m in keys:close(saved['totals'][m],fsum(v[m] for v in sums.values()),2e-14)
   costs={m:[d['model_and_reference_load_cpu'],d['face_setup']['cpu'],d['rules'][str(order)]['setup_cpu']] for m in ('zero','first','second','direct','actual')}
   for i in unique:
    row=rows[i];c=row['cpu']
    for m in costs:costs[m]+=[c['face']]+([c['frozen_partition']] if m=='zero' else [c['actual_partition']] if m=='actual' else [c['actual_partition'],c['frozen_partition']])
    for h in row['halves']:
     c=h['cpu'];costs['zero'] += [c['frozen_errors'],c['frozen_mass']];costs['actual'] += [c['actual_errors'],c['actual_mass']]
     for m in ('first','second','direct'):costs[m]+=[c['actual_errors'],c['frozen_errors'],c['matching'],c['frozen_mass'],c[m]]
   assert set(saved['metrics'])==set(costs)
   for m,terms in costs.items():
    sm=saved['metrics'][m];close(sm['composed_cpu'],fsum(terms));accuracy=1-saved['totals'][m];assert sm['accuracy']==accuracy
    delta=[F.from_float(accuracy)-F.from_float(x) for x in (reflo,refhi)];lo=F(0) if delta[0]*delta[1]<=0 else min(map(abs,delta));hi=max(map(abs,delta));cpu=F.from_float(sm['composed_cpu'])
    assert sm['absolute_error_interval']==list(map(float,(lo,hi))) and sm['cpu_squared_error_interval']==[float(cpu*lo*lo),float(cpu*hi*hi)]
    if m!='zero':assert sm['robust_halving_vs_zero_on_recorded_times']==(sm['cpu_squared_error_interval'][1]<.5*saved['metrics']['zero']['cpu_squared_error_interval'][0])
   metrics_out.append(saved)
 cap();r.update(complete=True,pass_all=True,producer_campaign_complete=complete,producer_expected_projection_stop=not complete,points_checked=len(rows),recurrences_checked=recurrences,mask_assignments=assignments,events=len(expected_events),summary=metrics_out,author_cpu=d['cpu_seconds'],source_face_calls_replayed=0,cdf_table_or_reference_replayed=False,direct_gaussian_values_recomputed=False,peak_bits_or_ledger_replayed=False,input_pins=ci['pins'])
if __name__=='__main__':
 assert not (P/'results.json').exists() and not (P/'results.tmp').exists();r=dict(complete=False,pass_all=False,points_checked=0,recurrences_checked=0);save(r)
 try:main(r)
 except BaseException:r.update(complete=False,pass_all=False,failure=traceback.format_exc());raise
 finally:save(r)
