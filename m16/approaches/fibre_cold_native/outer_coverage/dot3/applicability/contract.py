"""Pinned metadata, AST and saved-ledger contraction; no NumPy/model calls."""
from pathlib import Path
from collections import Counter
import ast,json,hashlib,struct,math,time,traceback,resource
P=Path(__file__).resolve().parent
NORM={
 '_subdivide:L294:np.linalg.norm(ab)':('_subdivide',294,'np.linalg.norm(ab)'),
 '_subdivide:L294:np.linalg.norm(bc)':('_subdivide',294,'np.linalg.norm(bc)'),
 '_subdivide:L294:np.linalg.norm(ca)':('_subdivide',294,'np.linalg.norm(ca)'),
 'cells_on_face:L342:np.linalg.norm(w)':('cells_on_face',342,'np.linalg.norm(w)'),
 'clip:L41:np.linalg.norm(n)':('clip',41,'np.linalg.norm(n)'),
 'clip:L62:np.linalg.norm(p)':('clip',62,'np.linalg.norm(p)'),
 'face_err:L107:np.linalg.norm(w)':('face_err',107,'np.linalg.norm(w)'),
 'face_err:L113:np.linalg.norm(w0)':('face_err',113,'np.linalg.norm(w0)')}
DOT='poly_int:L324:np.dot(v0, np.cross(v1, v2))';CLIP='clip:L42:n @ v'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def leaves(x):
 if isinstance(x,list):
  for y in x:yield from leaves(y)
 else:yield x

def main():
 out=P/'results.json';assert not out.exists(),'immutable attempt already exists';pr=read(P/'protocol.json');t=time.process_time();d=dict(complete=False,protocol_sha256=sha(P/'protocol.json'),source_sha256=sha(__file__))
 def save():
  d['cpu']=time.process_time()-t;d['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;out.write_text(json.dumps(d,indent=2)+'\n')
 save()
 try:
  for path,digest in pr['pins'].items():assert sha(path)==digest,path
  identity=read(pr['identity']);v=list(leaves(identity['model']));assert len(v)==104 and all(type(x) is float and math.isfinite(x) for x in v);assert identity['shapes']==[[8],[8,8],[4,8]]
  digest=hashlib.sha256(b''.join(struct.pack('<d',x) for x in v)).hexdigest();assert digest==identity['model_sha256'];d['model_metadata']=dict(parameters=len(v),all_finite_float_leaves=True,shape=identity['shapes'],model_sha256=digest)
  trees={name:ast.parse(Path(path).read_text()) for name,path in pr['numerical_sources'].items()};sites={**NORM,DOT:('poly_int',324,'np.dot(v0, np.cross(v1, v2))'),CLIP:('clip',42,'n @ v')};source_checks=[]
  for site,(fn,line,expr) in sites.items():
   file='fibre4b.py' if fn=='face_err' else 'fibre4.py';function=next(n for n in trees[file].body if isinstance(n,ast.FunctionDef) and n.name==fn)
   matches=[n for n in ast.walk(function) if isinstance(n,(ast.Call,ast.BinOp)) and n.lineno==line and ast.unparse(n)==expr];assert len(matches)==1,(site,len(matches));source_checks.append(dict(site=site,file=file,ast_expression=expr))
  census=read(pr['census']);assert census['complete'];counts=census['native_sites'];selected=[]
  norm_sites={s for s,c in counts.items() if 'norm_mul' in c};assert norm_sites==set(NORM)
  for site in NORM:
   c=counts[site];assert set(c)=={'norm_mul','norm_add','special_sqrt'};n=c['special_sqrt'];assert c['norm_mul']==3*n and c['norm_add']==2*n;selected.append(dict(site=site,calls=n,old_units=6*n,new_upper=8*n,kind='norm3'))
  for site,keys in [(DOT,('dot_mul','dot_add')),(CLIP,('matrix_mul','matrix_add'))]:
   c=counts[site];assert set(c)==set(keys) and c[keys[0]]%3==0;n=c[keys[0]]//3;assert c[keys[1]]==2*n;selected.append(dict(site=site,calls=n,old_units=5*n,new_upper=8*n,kind='dot3_upper'))
  norm_calls=sum(x['calls'] for x in selected if x['kind']=='norm3');assert norm_calls==census['by_name']['np.linalg.norm']['calls'];assert next(x['calls'] for x in selected if x['site']==DOT)==census['by_name']['np.dot']['calls']
  old=sum(x['old_units'] for x in selected);new=sum(x['new_upper'] for x in selected);prior=census['prior_hybrid_named_composition'];assert prior==1838695310
  control=read(pr['controls']);assert control['complete'] and control['control_count']==90 and control['all_bitwise_parity'] and control['restored']
  d.update(complete=True,source_checks=source_checks,selected=selected,selected_old=old,selected_new_numeric_upper=new,added_upper=new-old,prior_hybrid_named=prior,conditional_revised_hybrid=prior-old+new,source_invariant='Pinned real float64 model/families and closure under listed3vector operations, independently reviewable in PROOF.md.',historical_binary_binding_is_conditional=True,zero_model_or_kernel_evaluations=True,scope='Replace-once selected numeric upper only. All other matrix/native/libm/cold and integer/control/copy/allocation/runtime categories retain previous qualifications. No cost-ratio or full-native/hardware conclusion. Source and all-stride proof require independent acceptance.')
  save();print(json.dumps({k:d[k] for k in ('complete','selected_old','selected_new_numeric_upper','added_upper','conditional_revised_hybrid','cpu')}))
 except Exception:
  d['failure']=traceback.format_exc();save();raise
if __name__=='__main__':main()
