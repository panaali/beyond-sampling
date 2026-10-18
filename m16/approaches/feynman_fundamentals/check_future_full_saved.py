"""Independent raw bindings, closed coefficient maps and all-chart saved groups."""
from check_future_h8_prefix_saved import P,H,sha,frac,mat
from pathlib import Path
from fractions import Fraction as F
from collections import Counter,defaultdict
import json,math,time
def load(p):return json.loads(Path(p).read_text())
start=time.process_time();d=load(P/'full.json');proto=load(P/'full_protocol.json');initial=load(P.parent/'capture/zero_controls.json')['initial_cones']['4']
assert d['complete'] and d['protocol_sha256']==sha(P/'full_protocol.json')
for path,h in proto['pins'].items():assert sha(path)==h,path
results=[]
for model in d['models']:
 width=model['H'];assert width in (4,8);rows=model['rows'];sources={};parentrows=[];obs=Counter();newformation=Counter();cache={};exact_equal=0;maxdiff=F(0)
 for j,entry in enumerate(model['inputs']):
  path=Path(entry['path']);assert sha(path)==entry['sha256'];src=load(path);sources[str(path)]=src
  assert src['complete'] and src['checkpoint_sha256']==model['checkpoint_sha256']==proto['models'][str(width)]['checkpoint_sha256']
  assert entry['reused']==(j==0) and entry['pieces']==len(src['states'])==len(src['analysis']['rows'])
  if j==0:weights=src['model'];assert [x['parent'] for x in src['parents']]==[0,1,20,21]
  else:
   assert src['model']==weights and src['protocol_sha256']==d['protocol_sha256'] and src['batch']==j-1
   assert [x['parent'] for x in src['parents']]==proto['chunks'][j-1]
   progress=[json.loads(l) for l in path.with_name(path.stem+'_progress.jsonl').read_text().splitlines()]
   assert [e['row'] for e in progress if e['stage']=='prefix_state']==src['states']
   assert [e['row'] for e in progress if e['stage']=='state_result']==src['analysis']['rows']
   assert [e['index'] for e in progress if e['stage']=='state_attempt']==list(range(len(src['states'])))
   assert progress[-1]['stage']=='complete'
   unique=len({tuple(tuple(g) for g in x['history']) for x in src['states']})
   expected={k:unique*3*width*3*(width+1) for k in ('mul','add')}
   assert src['analysis']['formation_counts']==expected;newformation.update(expected)
  parentrows.extend(src['parents'])
  for k in ('proxy','finish_calls','lookahead_calls','lookahead_points','states','mass_or_final_calls'):obs[k]+=src['observer'][k]
 assert len(sources)==9 and len(parentrows)==96 and sorted(x['parent'] for x in parentrows)==list(range(96))
 assert parentrows==model['parents'] and dict(obs)==model['combined_observer'] and obs['mass_or_final_calls']==0
 assert dict(newformation)==model['new_formation_counts']
 wi=tuple(map(frac,weights[0]));wh=mat(weights[1]);groups={k:defaultdict(list) for k in model['groups']};parents=Counter();same=defaultdict(list)
 for i,r in enumerate(rows):
  src=sources[r['source']];s=src['states'][r['source_index']];analysis=src['analysis']['rows'][r['source_index']]
  assert r['index']==i and r['occurrence_id']==[width,s['parent'],r['source_index']]
  for k in ['parent','truth','history','exact_state','stored_state','map_equal','max_map_difference']:assert r[k]==analysis[k]
  hist=tuple(tuple(g) for g in s['history']);assert len(hist)==3 and all(len(g)==width and set(g)<=set((0,1)) for g in hist)
  g1,g2,g3=hist;assert s['depth']==3 and s['g']==list(g3) and s['truth']==initial[s['parent']]['truth']
  assert list(g1)==[int(initial[s['parent']]['sigma']*w>0) for w in wi]
  if hist not in cache:
   first=[g1[k]*wi[k] for k in range(width)];second=[g2[k]*sum(wh[k][j]*first[j] for j in range(width)) for k in range(width)]
   col0=[g3[k]*sum(wh[k][j]*second[j] for j in range(width)) for k in range(width)]
   col1=[g3[k]*sum(wh[k][j]*g2[j]*wi[j] for j in range(width)) for k in range(width)]
   cache[hist]=tuple((col0[k],col1[k],g3[k]*wi[k]) for k in range(width))
  exact=cache[hist];stored=tuple(tuple(F(float.fromhex(v)) for v in a[:3]) for a in s['state'])
  assert mat(r['exact_state'])==exact and mat(r['stored_state'])==stored
  for k,(a,b) in enumerate(zip(s['A'],s['state'],strict=True)):
   assert len(a)==len(b)==4 and all(math.isfinite(float.fromhex(v)) for v in a+b)
   assert all(float.fromhex(b[j])==g3[k]*float.fromhex(a[j]) for j in range(4)) and float.fromhex(a[3])==float.fromhex(b[3])==0
  difference=max(abs(x-y) for a,b in zip(exact,stored) for x,y in zip(a,b));assert frac(r['max_map_difference'])==difference and r['map_equal']==(exact==stored)
  maxdiff=max(maxdiff,difference);exact_equal+=exact==stored;parents[s['parent']]+=1;same[(s['parent'],hist)].append(i)
  for field,key in [('history',hist),('exact_state',exact),('stored_state',stored)]:
   groups['global_'+field][key].append(i);groups['same_truth_'+field][(s['truth'],key)].append(i)
 groups={k:list(v.values()) for k,v in groups.items()};assert groups==model['groups']
 assert parents==Counter({r['parent']:r['states'] for r in parentrows})
 resets={k:[g for g in v if len({tuple(tuple(x) for x in rows[i]['history']) for i in g})>1] for k,v in groups.items() if k.endswith('_state')};assert resets==model['distinct_history_reset_groups']
 cross={k:[g for g in groups[k] if len({rows[i]['parent'] for i in g})>1] for k in ('global_history','same_truth_history')};assert cross==model['cross_chart_history_groups']
 summary=model['summary'];assert summary['prefix_pieces']==len(rows)==obs['states'] and summary['parents']==96 and summary['terminal_cells'] is None
 assert summary['group_counts']=={k:len(v) for k,v in groups.items()} and summary['reset_group_counts']=={k:len(v) for k,v in resets.items()}
 assert summary['cross_chart_history_group_counts']=={k:len(v) for k,v in cross.items()} and summary['same_parent_repeated_history_groups']==sum(len(v)>1 for v in same.values())==0
 assert summary['stored_maps_equal_exact']==exact_equal and frac(summary['max_map_difference'])==maxdiff
 for k,n in [('distinct_global_states_per_prefix_piece',len(groups['global_exact_state'])),('distinct_same_truth_states_per_prefix_piece',len(groups['same_truth_exact_state']))]:assert frac(summary[k])==F(n,len(rows))
 results.append({'H':width,'summary':summary,'inputs':model['inputs'],'closed_map_histories':len(cache),'raw_coefficient_rows':len(rows),'new_formation_counts':dict(newformation),'recorded_observer':dict(obs)})
assert [r['H'] for r in results]==[4,8]
out={'complete':True,'owner_commit':'4b952e61','full_sha256':sha(P/'full.json'),'protocol_sha256':sha(P/'full_protocol.json'),'models':results,'review_cpu':time.process_time()-start,'scope':'Recipe108 represented prefix pieces, not exact-enumeration terminal cells. All96chart coverage/model and once-only pilots; raw state bindings, exact closed3column maps, groups and named formation checked. No geometry or author recurrence replay, no true-domain, mass, convex-union or total-cost proof. Internal observer calls conserved from saved records only.'}
(H/'future_full_saved_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'complete':True,'models':[(r['H'],r['raw_coefficient_rows'],r['closed_map_histories']) for r in results],'review_cpu':out['review_cpu']}))
