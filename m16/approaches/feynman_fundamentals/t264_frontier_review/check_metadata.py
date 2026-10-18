"""Verify only T264 text blobs and artifact metadata; never decode data payloads."""
import hashlib,json,pathlib,subprocess
PEER=pathlib.Path('/workspace/work/recover-small_h2_56');B=PEER/'review/pruning_cost_frontier';O=pathlib.Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(PEER),*args])
for name in ['CONTRACT.md','README.md','manifest.json']:
 assert (B/name).read_bytes()==git('show','dd5afb28:review/pruning_cost_frontier/'+name)
m=json.loads((B/'manifest.json').read_text());assert len(m['source_records'])==4 and len(m['proposed_fixed_artifacts'])==3
rows=[];pins={str(B/n):sha(B/n) for n in ['CONTRACT.md','README.md','manifest.json']}
for r in m['source_records']:
 p=PEER/r['snapshot'];raw=p.read_bytes()
 assert raw==git('show',m['owner_commit']+':'+r['git_path']) and sha(p)==r['sha256']
 assert git('rev-parse',m['owner_commit']+':'+r['git_path']).decode().strip()==r['blob']
 rows.append({'path':r['git_path'],'blob':r['blob'],'sha256':r['sha256'],'passed':True});pins[str(p)]=sha(p)
artifacts=[]
for r in m['proposed_fixed_artifacts']:
 line=git('ls-tree','--full-tree','-l',m['owner_commit'],'--',r['git_path']).decode().strip()
 header,path=line.split('\t');mode,kind,blob,size=header.split()
 assert kind=='blob' and path==r['git_path'] and blob==r['blob'] and int(size)==r['bytes'] and r['payload_decoded'] is False
 artifacts.append({'path':path,'blob':blob,'bytes':int(size),'payload_read':False,'passed':True})
pins[str(O/'check_metadata.py')]=sha(O/'check_metadata.py')
out={'complete':True,'pass_all':True,'execution_allowed':False,'task':'T264','assignment':'d9b83886','reviewed_document_commit':'dd5afb28','owner_commit':m['owner_commit'],'source_records':rows,'artifact_metadata':artifacts,'pins':pins,'calls':{'author':0,'model':0,'geometry':0,'mass':0,'rng':0,'array_decode':0,'retained_value_optimization':0},'scope':'Written source/oracle theorem and exact text/object-metadata bindings only'}
assert not (O/'source_review.json').exists();(O/'source_review.json').write_text(json.dumps(out,indent=2)+'\n');print('four text bindings and three unopened artifact metadata entries pass')
