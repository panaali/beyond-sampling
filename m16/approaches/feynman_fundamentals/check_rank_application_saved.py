"""Saved128 exact rank witnesses; reuses independent reviewer scalar checks, no author imports."""
from check_rank_witness_saved import rank,determinant,bits,unpack,sha,H,F,Counter,itertools,json,time,P
def main():
 start=time.process_time();raw=json.loads((P/'application.json').read_text());d=unpack(raw);p=json.loads((P/'application_protocol.json').read_text());old=unpack(json.loads(__import__('pathlib').Path(p['trained']).read_text()))
 for f,h in p['pins'].items():assert sha(f)==h
 assert d['complete'] and d['protocol_sha256']==sha(P/'application_protocol.json') and d['inherited_trained_sha256']==sha(p['trained'])
 events=[json.loads(x) for x in (P/'application_progress.jsonl').read_text().splitlines()]
 assert Counter(e['stage'] for e in events)==dict(phase_attempt=1,attempt=128,result=128,complete=1)
 assert [e['row'] for e in events if e['stage']=='result']==raw['rows']
 assert [e['identity'] for e in events if e['stage']=='attempt']==[r['identity'] for r in raw['rows']]
 expected=[]
 for model in old['models']:
  for pair in model['rows']:
   for side,dom in enumerate(pair['result']['domains']):
    identity=dict(H=model['H'],truth=pair['truth'],rank=pair['rank'],pair_index=pair['pair_index'],side=side,occurrence_id=pair['a' if side==0 else 'b']['occurrence_id'])
    expected.append((identity,dom))
 assert len(expected)==len(d['rows'])==128
 totals=Counter();dettotals=Counter();decodes=0;summary={};ids=[]
 for row,(identity,dom) in zip(d['rows'],expected,strict=True):
  assert row['identity']==identity and row['inherited_status']==dom['status']
  linear,hinges=dom['canonical']['key'];zero=not hinges and not any(x for v in linear for x in v)
  assert not zero and not dom['certified'] and row['canonical_nonzero']
  R=[[F(x) for x in v] for v in dom['rays']];ans=row['result'];r=rank(R)
  assert r==ans['rank']==4 and row['status']=='certified_unequal_on_represented_hull'
  selected=[]
  for i in range(len(R)):
   if rank([R[j] for j in selected]+[R[i]])>len(selected):selected.append(i)
   if len(selected)==4:break
  assert selected==ans['selected_indices'];limit=selected[-1]+1 if r==4 else len(R)
  assert ans['total_ray_rows']==len(R) and ans['considered_ray_rows']==len(ans['trace'])==limit and ans['rational_decodes']==4*len(R)
  basis=[];pivots=[];counts=Counter();values=[]
  for i,record in enumerate(ans['trace']):
   z=R[i][:]
   for pivot,b in zip(pivots,basis,strict=True):
    c=z[pivot]
    if c:
     products=[c*x for x in b];values+=products;z=[x-y for x,y in zip(z,products,strict=True)];values+=z;counts['mul']+=4;counts['sub']+=4
   pivot=next((j for j,x in enumerate(z) if x),None);assert record['index']==i and record['residual']==z and record['pivot']==pivot
   if pivot is not None:
    b=[x/z[pivot] for x in z];assert record['normalized']==b;values+=b;counts['div']+=4;basis.append(b);pivots.append(pivot)
  assert ans['pivots']==pivots and ans['elimination_counts']==dict(counts) and ans['elimination_observed_bits']==bits(values)
  if r<4:assert ans['determinant'] is None and ans['status']=='unknown_rank_deficient'
  else:
   a=[R[j] for j in selected];det=ans['determinant'];assert det['value']==determinant(a)!=0 and ans['status']=='full_rank_witness'
   terms=[];values=[];total=F(0)
   for perm in itertools.permutations(range(4)):
    v=a[0][perm[0]]
    for i in range(1,4):v*=a[i][perm[i]];values.append(v)
    odd=bool(sum(perm[i]>perm[j] for i in range(4) for j in range(i+1,4))%2);total+=-v if odd else v;values.append(total);terms.append(dict(permutation=list(perm),odd=odd,product=v))
   assert det['terms']==terms and det['counts']==dict(mul=72,add=12,sub=12) and det['observed_bits']==bits(values) and total==det['value']
  assert row['selected_original_rays']==[R[j] for j in selected]
  totals.update(ans['elimination_counts']);dettotals.update(ans['determinant']['counts']);decodes+=ans['rational_decodes'];ids.append(tuple(identity[k] for k in ('H','truth','rank','side')))
 assert len(set(ids))==128 and set(ids)=={(h,t,k,s) for h in [4,8] for t in range(4) for k in range(8) for s in [0,1]}
 for h in [4,8]:summary[str(h)]=dict(hulls=64,statuses={'certified_unequal_on_represented_hull':64},ranks={'4':64})
 assert summary==d['summary'] and dict(totals)==d['elimination_arithmetic'] and dict(dettotals)==d['determinant_arithmetic'] and decodes==d['rational_decodes']
 out=dict(complete=True,pass_all=True,source_result_commit='482c6b86',pins={n:sha(P/n) for n in ['application.json','application_protocol.json','application_progress.jsonl']},checker_sha256=sha(__file__),rows=128,rank4_witnesses=128,nonzero_determinants=128,progress_events=len(events),summary=summary,review_cpu=time.process_time()-start,author_cpu=d['cpu_seconds'],elimination_arithmetic=dict(totals),determinant_arithmetic=dict(dettotals),rational_decodes=decodes,new_author_geometry_model_calls=0,scope='Exact saved binding/minor/trace/determinant/count audit; prior reviewed nonzero canonical keys reused. T208 gives function inequality on each represented hull only, not label difference or true domain feasibility. No numerical or geometry author replay.')
 (H/'rank_application_saved_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(complete=True,review_cpu=out['review_cpu'])))
if __name__=='__main__':main()
