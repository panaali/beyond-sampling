"""Independent text/tree/schema validation, never author or binary execution."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

B=Path('/workspace/work/feynman_fundamentals/m16/approaches/feynman_fundamentals/capacity_retained_evidence')
P=Path(__file__).resolve().parent
M=json.loads((B/'manifest.json').read_text()); rev=M['author_head']
tree=subprocess.check_output(['git','ls-tree','--full-tree','-rl',rev,'--','m16/approaches/capacity_ceiling/'],text=True)
assert tree==(B/'AUTHOR_TREE.txt').read_text()
entries={}
for line in tree.splitlines():
    metadata,path=line.split('\t'); mode,kind,blob,size=metadata.split()
    assert kind=='blob'; entries[path]=(blob,int(size))
assert len(entries)==21 and len(M['files'])==15 and len(M['unopened_binary_entries'])==6
pins={}
for row in M['files']:
    raw=(B/row['snapshot']).read_bytes()
    assert entries.pop(row['original_path'])==(row['blob'],row['size'])
    assert len(raw)==row['size'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    assert raw==subprocess.check_output(['git','show',rev+':'+row['original_path']])
    if row['snapshot'].endswith('.py'): ast.parse(raw)
    pins[str(B/row['snapshot'])]=row['sha256']
for row in M['unopened_binary_entries']:
    assert entries.pop(row['path'])==(row['blob'],row['size'])
    assert row['path'].endswith('.npz')
assert not entries
delta=subprocess.check_output(['git','diff','--name-only','d2fdf6bd',rev,'--','m16/approaches/capacity_ceiling/'],text=True).splitlines()
assert all(path.endswith('/README.md') or '/results/' in path or '/weights/' in path for path in delta)

# Validate against the saved schema recursively, rather than invoking its generator.
def conforms(value, schema):
    if isinstance(schema,str):
        assert type(value).__name__==schema
    elif schema.get('type')=='list' and set(schema)=={'type','length','member_schemas'}:
        assert type(value) is list and len(value)==schema['length']
        for item in value:
            accepted=False
            for candidate in schema['member_schemas']:
                try: conforms(item,candidate)
                except AssertionError: continue
                accepted=True;break
            assert accepted
    else:
        assert type(value) is dict and set(value)==set(schema)
        for key in value: conforms(value[key],schema[key])
schemas=json.loads((B/'SCHEMA.json').read_text())
assert set(schemas)=={'snapshot/results/a30.json','snapshot/results/zoo_code_seed0.json'}
for name,schema in schemas.items(): conforms(json.loads((B/name).read_text()),schema)
A=json.loads((B/'snapshot/results/a30.json').read_text())
Z=json.loads((B/'snapshot/results/zoo_code_seed0.json').read_text())
assert 'src_hash' not in A and len(A['curve'])==64 and len(A['eval'])==5
assert all(len(row['loss'])==len(row['acc'])==5 for row in A['curve'])
for row in [*A['eval'].values(),Z['eval']]:
    assert 'correct' not in row and 'discordant_correct_vs' not in row
for name in ['SOURCE_REVIEW.md','manifest.json','SCHEMA.json','AUTHOR_TREE.txt','check_metadata.py','metadata_result.json']:
    pins[str(B/name)]=hashlib.sha256((B/name).read_bytes()).hexdigest()
for name in ['README.md','check_metadata.py']:
    pins[str(P/name)]=hashlib.sha256((P/name).read_bytes()).hexdigest()
print(json.dumps(dict(task='T257',reviewer='recover_34',complete=True,pass_all=True,
    execution_allowed=False,execution_authorized=False,author_package='d9208ca4',owner_revision=rev,
    pins=pins,text_json_blobs_verified=15,binary_tree_entries_unopened=6,json_schemas=2,
    author_calls=0,binary_payload_reads=0,score_contractions=0,
    scope='Written source/schema/provenance only; no runtime, model identity, score or capacity validation'),indent=2))
