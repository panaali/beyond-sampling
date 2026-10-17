"""Saved768 exact affine-map contraction only; no trained network forward."""
import gzip,hashlib,json,resource,time,traceback
from collections import Counter
from pathlib import Path
from core import Arithmetic,reconstruct,analyse,pack
HERE=Path(__file__).resolve().parent
SOURCE=Path('/workspace/work/recover-small_h2_34/m16/approaches/fading_influence/trained.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 start=time.process_time();dst=HERE/'trained.json';raw=HERE/'trained_rows.jsonl.gz'
 assert not dst.exists() and not raw.exists()
 gate=json.loads((HERE/'trained_gate.json').read_text());assert gate['complete']
 for p,h in gate['pins'].items():assert sha(Path(p))==h
 control=json.loads((HERE/'controls.json').read_text());assert control['complete']
 source=json.loads(SOURCE.read_text());assert source['complete'] and len(source['models'])==2
 assert [(m['name'],len(m['witnesses'])) for m in source['models']]==[('H4T4seed0',384),('H8T4seed0',384)]
 out=dict(complete=False,gate_sha256=sha(HERE/'trained_gate.json'),source_sha256=sha(SOURCE),models=[],rows_completed=0,new_trained_forward_calls=0,new_reference_draws=0,raw=str(raw),scope='Finite saved exact-map common-mode discriminator; no probability, arbitrary-projection impossibility or global invariance claim.')
 def save():
  out['cpu']=time.process_time()-start;out['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  tmp=dst.with_suffix('.tmp');tmp.write_text(json.dumps(pack(out),indent=2)+'\n');tmp.replace(dst)
 save()
 try:
  with gzip.open(raw,'wt',encoding='utf8',compresslevel=3) as archive:
   for model in source['models']:
    assert model['complete'];m=model['model'];assert m['T']==4 and m['H']==(4 if model['name']=='H4T4seed0' else 8)
    summary=dict(name=model['name'],model=m,checkpoint_sha256=model['checkpoint_sha256'],complete=False,rows=0,robust_obstructions=0,line_open_changes=0,affine_probe_evaluations=0,first_obstruction_index=None,counts={},max_integer_bits=0)
    out['models'].append(summary);total=Counter()
    for i,w in enumerate(model['witnesses']):
     if time.process_time()-start>16:raise RuntimeError('Preserved bounded attempt:16CPU graceful limit before next row')
     assert w['index']==i and w['strict'] and w['local_derivative_certified'] and not w['boundary_gates']
     A=Arithmetic();r=reconstruct(m,w['x'],w['history'],A)
     assert r['L']==w['logit_map_numerators'] and r['denominator']==w['logit_denominator']
     nonzero={(g['t'],g['i']) for g in r['gate_rows']}
     zeros=[[t,j] for t in range(m['T']) for j in range(m['H']) if (t,j) not in nonzero]
     assert zeros==w['identically_zero_gates']
     result=analyse(w['x'],r,A)
     row=dict(model=model['name'],index=i,x=w['x'],history=w['history'],saved_radius=w['radius'],reconstructed=r,result=result)
     archive.write(json.dumps(pack(row),separators=(',',':'))+'\n');archive.flush()
     summary['rows']+=1;out['rows_completed']+=1
     if result['robust_opposite_pair'] is not None:
      summary['robust_obstructions']+=1
      if summary['first_obstruction_index'] is None:summary['first_obstruction_index']=i
     summary['line_open_changes']+=int(result['line_open_change'])
     summary['affine_probe_evaluations']+=len(result['open_probes'])+len(result['crossing_points'])
     total.update(result['counts']);summary['counts']=dict(total);summary['max_integer_bits']=max(summary['max_integer_bits'],result['max_integer_bits']);save()
    assert summary['rows']==384;summary['complete']=True;save()
  assert out['rows_completed']==768
  out.update(complete=True,raw_sha256=sha(raw));save()
  print(json.dumps(dict(complete=True,cpu=out['cpu'],models=[{k:m[k] for k in ('name','rows','robust_obstructions','line_open_changes','affine_probe_evaluations')} for m in out['models']])))
 except Exception as exc:
  out['failure']=dict(error=repr(exc),traceback=traceback.format_exc());out['partial_raw_sha256']=sha(raw) if raw.exists() else None;save();raise
if __name__=='__main__':main()
