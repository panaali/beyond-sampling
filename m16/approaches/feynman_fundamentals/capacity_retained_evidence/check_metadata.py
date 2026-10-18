"""T257 source-byte/schema inventory only; no score arithmetic or binary decoding."""
from pathlib import Path
import hashlib,json,subprocess,time
P=Path(__file__).resolve().parent
start=time.process_time()
def schema(x):
    if isinstance(x,dict):return {k:schema(v) for k,v in x.items()}
    if isinstance(x,list):
        shapes={json.dumps(schema(v),sort_keys=True) for v in x}
        return {'type':'list','length':len(x),'member_schemas':[json.loads(t) for t in sorted(shapes)]}
    return type(x).__name__
m=json.loads((P/'manifest.json').read_text())
assert len(m['files'])==15 and len(m['unopened_binary_entries'])==6
for r in m['files']:
    b=(P/r['snapshot']).read_bytes()
    assert hashlib.sha256(b).hexdigest()==r['sha256'] and len(b)==r['size']
    assert subprocess.check_output(['git','show',m['author_head']+':'+r['original_path']])==b
    assert subprocess.check_output(['git','rev-parse',m['author_head']+':'+r['original_path']],text=True).strip()==r['blob']
tree=subprocess.check_output(['git','ls-tree','-r','--long',m['author_head'],'m16/approaches/capacity_ceiling/'],text=True)
assert tree==(P/'AUTHOR_TREE.txt').read_text()
for r in m['unopened_binary_entries']:
    assert r['path'].endswith('.npz')
    assert any(('blob '+r['blob']) in line and line.endswith('\t'+r['path']) for line in tree.splitlines())
saved=json.loads((P/'SCHEMA.json').read_text())
assert saved=={r['snapshot']:schema(json.loads((P/r['snapshot']).read_text())) for r in m['files'] if r['snapshot'].endswith('.json')}
res={'complete':True,'pass_all':True,'text_bindings':15,'json_schemas':2,'binary_entries_unopened':6,'schema_method':'keys/types/list lengths only; scalar score values not contracted','author_head':m['author_head'],'cpu_metadata':time.process_time()-start,'author_model_training_rng_calls':0,'binary_payload_reads':0,'score_arithmetic':0}
with (P/'metadata_result.json').open('x') as f:json.dump(res,f,indent=2);f.write('\n')
