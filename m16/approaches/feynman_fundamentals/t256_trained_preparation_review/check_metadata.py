"""Written-plan metadata only; no trained payload decoding or author imports."""
import hashlib,json,pathlib,subprocess
B=pathlib.Path('/workspace/work/recover-small_h2_78/review/gap_box_affine_range/trained_preparation')
O=pathlib.Path(__file__).resolve().parent
files=['PLAN.md','selection.json','provenance.json','README.md','CORRECTION.md']
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
pins={str(B/x):sha(B/x) for x in files}
for name in files:
 original=subprocess.check_output(['git','-C',str(B),'show','b3fd6c85:review/gap_box_affine_range/trained_preparation/'+name])
 assert original==(B/name).read_bytes()
s=json.loads((B/'selection.json').read_text());p=json.loads((B/'provenance.json').read_text())
assert not s['execution_authorized'] and s['proposed_new_calls']=={'whole_box':4,'restricted':4}
assert s['boundaries_v']==[-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8] and len(s['rows'])==4
rows=[]
for i,r in enumerate(s['rows']):
 bins=[8,7,6,5];bins=bins[-i:]+bins[:-i] if i else bins
 assert r['bin_ids']==bins and r['leader']==i and r['second']==(i+1)%4 and r['physical_gap']=='1/4'
 index=sum(j*12**(3-k) for k,j in enumerate(bins));chunk,offset=divmod(index,256)
 assert (index,chunk,offset)==(r['index'],r['chunk'],r['offset'])
 assert r['box_v']==[[s['boundaries_v'][j],s['boundaries_v'][j+1]] for j in bins]
 rows.append({'name':r['name'],'index':index,'chunk':chunk,'offset':offset,'passed':True})
assert sha(p['prior_review'])==p['prior_review_sha256'] and sha(p['prior_scope'])==p['prior_scope_sha256']
prior=json.loads(pathlib.Path(p['prior_review']).read_text());scope=json.loads(pathlib.Path(p['prior_scope']).read_text())
assert prior['complete'] and prior['run_scope_sha256']==p['prior_scope_sha256']
assert len(p['selected_payloads'])==6
for r in p['selected_payloads']:
 path=pathlib.Path(r['path']);assert path.stat().st_size==r['bytes'] and sha(path)==r['sha256'] and scope['pins'][str(path)]==r['sha256']
 pins[str(path)]=r['sha256']
for k in ['prior_scope','prior_review']:pins[p[k]]=sha(p[k])
for name in ['application.py','enclosure.py']:
 path=B.parents[2]/'m16/approaches/conditional_order_bounds/local_certificate'/name
 # B.parents[2] is the peer worktree root.
 assert str(path) in scope['pins'] and sha(path)==scope['pins'][str(path)];pins[str(path)]=sha(path)
pins[str(O/'check_metadata.py')]=sha(O/'check_metadata.py')
r={'complete':True,'pass_all':True,'execution_allowed':False,'task':'T256','assignment':'7fa65c98','producer_source':'8a9846b3','written_correction':'b3fd6c85','rows':rows,'pins':pins,'trained_payload_decodes':0,'candidate_calls':0,'model_calls':0,'probability_calls':0,'scope':'Written unit/selection/domain/provenance prerequisite only; future source/resource/execution gates still required'}
assert not (O/'source_review.json').exists();(O/'source_review.json').write_text(json.dumps(r,indent=2)+'\n');print('four selections and six byte bindings pass')
