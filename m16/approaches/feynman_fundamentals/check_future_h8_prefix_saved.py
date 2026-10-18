"""Exact saved H8 prefix coefficient/group audit without author imports."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter,defaultdict
import json,hashlib,math,time
H=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/future_equivalence/full_count')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frac(x):return F(int(x[0]),int(x[1]))
def mat(x):return tuple(tuple(frac(v) for v in row) for row in x)
def main():
 start=time.process_time();p=json.loads((P/'protocol.json').read_text());d=json.loads((P/'pilot.json').read_text());events=[json.loads(s) for s in (P/'pilot_progress.jsonl').read_text().splitlines()]
 assert d['complete'] and 'failure' not in d and d['protocol_sha256']==sha(P/'protocol.json')
 for f,h in p['pins'].items():assert sha(f)==h,f
 assert p['source_gate']!='pending' and p['parents']==[0,1,20,21]
 archive=json.loads(Path(p['model_archive']).read_text());m=next(r for r in archive['models'] if r['name']=='H8T4seed0');raw=m['model'];D=raw['D'];wi=tuple(F(x,D) for x in raw['wi']);wh=tuple(tuple(F(v,D) for v in row) for row in raw['wh']);wo=tuple(tuple(F(v,D) for v in row) for row in raw['wo'])
 assert tuple(map(frac,d['model'][0]))==wi and mat(d['model'][1])==wh and mat(d['model'][2])==wo
 assert d['checkpoint_sha256']==m['checkpoint_sha256']==p['checkpoint_sha256'] and d['rational_weight_decodes']==104
 predicate=json.loads(Path(p['predicate_review']).read_text());pr=next(r for r in predicate['rows'] if r['name']=='H8T4seed0');assert predicate['complete'] and all(pr['checks'].values()) and pr['subsets']==256
 zero=json.loads((P.parent/'capture/zero_controls.json').read_text());initial=zero['initial_cones']['4'];assert zero['all_pass'] and len(initial)==96
 states=d['states'];rows=d['analysis']['rows'];assert len(states)==len(rows)==81
 assert [r['row'] for r in events if r['stage']=='prefix_state']==states
 assert [r['row'] for r in events if r['stage']=='state_result']==rows
 assert [r['index'] for r in events if r['stage']=='state_attempt']==list(range(81))
 assert [r['parent'] for r in events if r['stage']=='parent_attempt']==[0,1,20,21]
 assert [r['row'] for r in events if r['stage']=='parent_result']==d['parents']
 assert Counter(r['stage'] for r in events)=={'phase_attempt':1,'model_binding':1,'parent_attempt':4,'prefix_state':81,'parent_result':4,'state_attempt':81,'state_result':81,'complete':1}
 groups={k:defaultdict(list) for k in d['analysis']['groups']};unique=set();maxdiff=F(0);equal=0;parent_counts=Counter()
 for i,(s,r) in enumerate(zip(states,rows,strict=True)):
  hist=tuple(tuple(g) for g in s['history']);assert len(hist)==3 and all(len(g)==8 and all(v in (0,1) for v in g) for g in hist)
  g1,g2,g3=hist;h=initial[s['parent']];assert s['truth']==h['truth'] and s['depth']==3 and s['g']==list(g3)
  assert list(g1)==[int(h['sigma']*x>0) for x in wi]
  # Closed three-column coefficients, independent of author's generic recurrence loop.
  first=[g1[k]*wi[k] for k in range(8)]
  step2=[g2[k]*sum(wh[k][l]*first[l] for l in range(8)) for k in range(8)]
  col0=[g3[k]*sum(wh[k][l]*step2[l] for l in range(8)) for k in range(8)]
  col1=[g3[k]*sum(wh[k][l]*g2[l]*wi[l] for l in range(8)) for k in range(8)]
  exact=tuple((col0[k],col1[k],g3[k]*wi[k]) for k in range(8))
  stored=tuple(tuple(F(float.fromhex(x)) for x in row[:3]) for row in s['state'])
  assert len(s['A'])==len(s['state'])==8
  for k,(a,b) in enumerate(zip(s['A'],s['state'],strict=True)):
   assert len(a)==len(b)==4
   assert all(math.isfinite(float.fromhex(v)) for v in a+b)
   assert all(float.fromhex(b[j])==g3[k]*float.fromhex(a[j]) for j in range(4))
   assert float.fromhex(a[3])==float.fromhex(b[3])==0
  for field in ('R','N'):
   assert all(len(x)==4 and all(math.isfinite(float.fromhex(v)) for v in x) for x in s['cone'][field])
  assert r['index']==i and r['occurrence_id']==[s['parent'],i] and r['history']==s['history']
  for field in ('parent','truth','depth'):assert r[field]==s[field]
  assert mat(r['exact_state'])==exact and mat(r['stored_state'])==stored
  difference=max(abs(x-y) for a,b in zip(exact,stored) for x,y in zip(a,b));assert frac(r['max_map_difference'])==difference and r['map_equal']==(stored==exact)
  maxdiff=max(maxdiff,difference);equal+=stored==exact;unique.add(hist);parent_counts[s['parent']]+=1
  for field,value in [('history',hist),('exact_state',exact),('stored_state',stored)]:
   groups['global_'+field][value].append(i);groups['same_truth_'+field][(s['truth'],value)].append(i)
 resultgroups={k:list(v.values()) for k,v in groups.items()};assert resultgroups==d['analysis']['groups']
 resets={k:[g for g in gs if len({tuple(tuple(x) for x in states[i]['history']) for i in g})>1] for k,gs in resultgroups.items() if k.endswith('_state')}
 assert resets==d['analysis']['distinct_history_reset_groups']
 summary={'prefix_pieces':81,'group_counts':{k:len(v) for k,v in resultgroups.items()},'reset_group_counts':{k:len(v) for k,v in resets.items()},'stored_maps_equal_exact':equal,'max_map_difference':[str(maxdiff.numerator),str(maxdiff.denominator)]};assert summary==d['analysis']['summary']
 assert len(unique)==d['analysis']['unique_history_reconstructions']==42 and d['analysis']['cached_history_reuses']==39
 # Generic author's formation loop executes 3*8*3*(8+1) of each operation per new history.
 counts={'mul':42*3*8*3*9,'add':42*3*8*3*9};assert d['analysis']['formation_counts']==counts
 assert [r['parent'] for r in d['parents']]==[0,1,20,21]
 for r in d['parents']:assert r['T']==4 and r['states']==parent_counts[r['parent']]
 obs=d['observer'];assert obs['states']==sum(r['states'] for r in d['parents'])==81 and obs['proxy']==sum(r['source_proxy'] for r in d['parents'])==198099
 assert obs['mass_or_final_calls']==d['new_mass_reference_MC_calls']==d['new_initial_cones']==0 and d['reused_initial_cones']==4
 assert events[-1]=={'stage':'complete','summary':summary}
 result={'complete':True,'source_artifact_commit':'b1802229','pilot_sha256':sha(P/'pilot.json'),'protocol_sha256':sha(P/'protocol.json'),'checker_sha256':sha(Path(__file__)),'exact_coefficient_rows':81,'summary':summary,'formation_counts':counts,'recorded_observer':obs,'saved_event_count':len(events),'author_cpu':d['cpu_seconds'],'review_cpu':time.process_time()-start,'new_forward_geometry_mass_calls':0,'limitations':['Four fixed parents only; prefix pieces are not terminal cells or probability weights.','Exact states refer to frozen histories; no proof of true recipe-domain feasibility or clipping coverage.','Identifiability premise reuses independent T11597 exact weight review; no new canonicalization run.','Proxy totals conserved from parents; internal observer finish/lookahead counts not independently regenerated.','Stored binary states differ from exact states in all81; equality-class counts agree, not map identity.','Counted formation excludes comparisons/GCD/memory/integer internals and capture setup; no full cost or speed claim.']}
 (H/'future_h8_prefix_saved_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('complete','exact_coefficient_rows','saved_event_count','review_cpu')}))
if __name__=='__main__':main()
