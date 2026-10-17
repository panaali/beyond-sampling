"""Saved byte/source/row bindings only; no model reconstruction or evaluator."""
from pathlib import Path
import hashlib,json,time,traceback,sys
HERE=Path(__file__).resolve().parent

def main():
 out=HERE/'provenance.json';assert not out.exists();start=time.process_time()
 gate=json.loads((HERE/'provenance_gate.json').read_text())
 for name,pin in gate.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==pin,name
 manifest=json.loads((HERE/'manifest.json').read_text())
 for name,pin in manifest['pins'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==pin,name
 native=Path(manifest['native_root']);recovery=Path(manifest['source_root'])
 binding=json.loads(Path(manifest['prior_trace_binding']).read_text());assert binding['complete'] and len(binding['rows'])==64
 assert binding['trace_sha256']==manifest['pins'][str(native/'trace.py')]
 assert binding['source_sha256']==manifest['pins'][str(recovery/'dependencies/understand_m43/est75.py')]
 inventory=json.loads((native/'inventory.json').read_text())
 for path,pin in inventory['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==pin,path
 cps={(c['H'],c['zoo']):c for c in manifest['checkpoints']};assert len(cps)==15
 raw={h:json.loads((recovery/'snapshot/out'/f'fresh_H{h}_abs0.05_80-81-82-83.json').read_text()) for h in (4,5,6)}
 rows=[];seen=set();model_hashes=set()
 for spec in manifest['shards']:
  h,z,n=spec['H'],spec['zoo'],spec['noise'];key=(h,z,n);assert key not in seen;seen.add(key)
  snap=HERE/spec['snapshot'];assert hashlib.sha256(snap.read_bytes()).hexdigest()==spec['sha256']
  old=json.loads(snap.read_text());assert old['complete'] and old['all_parity_pass'] and len(old['rows'])==old['expected']==64
  assert (old['H'],old['zoo'],old['noise'],old['borderline'])==(h,z,n,False)
  assert old['checkpoint_sha256']==cps[(h,z)]['sha256']
  rp=recovery/'snapshot/out'/f'fresh_H{h}_abs0.05_80-81-82-83.json';assert old['source_sha256']==manifest['pins'][str(rp)]
  panel=raw[h];assert panel['H']==h and panel['mode']=='abs' and panel['sigma']==.05 and panel['n']==64 and panel['seeds']==[80,81,82,83]
  original=panel['zoo'][str(z)]['rows'];assert len(original)==257
  for idx,r in enumerate(old['rows']):
   assert r['copy']==idx and r['parity_pass'] and r['cell_count_equal']
   o=original[1+64*panel['seeds'].index(n)+idx]
   for a,b in [('truth','truth'),('saved_estimate','est'),('saved_ops','ops'),('saved_cells','cells')]:assert r[a]==o[b],(key,idx,a)
   digest=r['model_sha256'];assert len(digest)==64 and all(x in '0123456789abcdef' for x in digest);assert digest not in model_hashes;model_hashes.add(digest)
  rows.append({'H':h,'zoo':z,'noise':n,'copies':64,'snapshot_sha256':spec['sha256'],'checkpoint_sha256':cps[(h,z)]['sha256'],'original_rows_bound':True})
 assert len(rows)==60 and len(model_hashes)==3840 and seen=={(h,z,n) for h in (4,5,6) for z in range(5) for n in (80,81,82,83)}
 result={'complete':True,'shards':rows,'cases':len(model_hashes),'checkpoints':manifest['checkpoints'],'manifest_sha256':gate['manifest.json'],'driver_sha256':gate['provenance.py'],
  'trace_binding':manifest['prior_trace_binding'],'independent_model_hash_regeneration_reused':'88490d79','current_model_regenerations':0,'current_estimator_reference_MC_calls':0,'cpu_seconds':time.process_time()-start,
  'scope':'Byte/source/saved-row bindings only; historical truth/reference qualification and prior independent hash regeneration are inherited, not new numerical validation.'}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['complete','cases','current_model_regenerations','current_estimator_reference_MC_calls','cpu_seconds']}))
if __name__=='__main__':
 try:main()
 except Exception:
  (HERE/'provenance_failure.json').write_text(json.dumps({'traceback':traceback.format_exc()},indent=2)+'\n');raise
