"""Separate-process exact-order cell continuation, frozen components6 then12."""
import ast,hashlib,importlib.util,json,resource,signal,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;inv=json.loads((HERE/'inventory.json').read_text());idx=int(sys.argv[2]);assert idx in (6,12)
for path,sha in inv['files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
native=Path(inv['native']);prior=Path(inv['prior']);folder=HERE/'cell_rescue'/f'{idx:02d}';folder.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('cell_t180_tracer',native/'tracer.py');T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
model=tuple(np.array(w) for w in json.loads((prior/'noise_identity.json').read_text())['model']);expected=json.loads((prior/'noise_components'/f'{idx:02d}.json').read_text());assert expected['complete']
assert hashlib.sha256(b''.join(w.astype('<f8').tobytes() for w in model)).hexdigest()==expected['model_sha256']==inv['model_sha256']

def cap(*_):raise TimeoutError('Cell rescue OS soft105/hard110 CPU limit')
resource.setrlimit(resource.RLIMIT_CPU,(105,110));signal.signal(signal.SIGXCPU,cap)
mode=sys.argv[1];source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
if mode in ('prepare','cell'):
 path=folder/('prepare.json' if mode=='prepare' else f'cell_{int(sys.argv[3]):05d}.json');assert not path.exists(),'Preserve previous attempts'
 out={'complete':False,'component':idx,'mode':mode,'source_sha256':source_sha,'model_sha256':inv['model_sha256'],'inventory_sha256':hashlib.sha256((HERE/'inventory.json').read_bytes()).hexdigest()};path.write_text(json.dumps(out,indent=2)+'\n');start=time.process_time();ledger=None
 try:
  FX,F3,Ops,ledger=T.load(True);ops=Ops();fam=FX.FAMS4[expected['family']];side=fam['sides'][expected['side']]
  if mode=='prepare':
   cells=[]
   for poly_idx,poly in enumerate(fam['polys']):
    for cell in FX.cells_on_face(model,poly,fam['B'],ops):
     assert all(v.dtype==np.float64 and v.shape==(3,) for v in cell)
     cells.append({'poly':poly_idx,'vertices':[v.tolist() for v in cell]})
   out.update(cells=cells,cell_count=len(cells),clamps=len(side[2]),legacy_ops=ops.n)
  else:
   k=int(sys.argv[3]);prep_path=folder/'prepare.json';prep=json.loads(prep_path.read_text());assert prep['complete'] and prep['source_sha256']==source_sha and 0<=k<prep['cell_count']
   resume=0.0;previous_sha=None
   if k:
    previous_path=folder/f'cell_{k-1:05d}.json';prev=json.loads(previous_path.read_text());assert prev['complete'] and prev['source_sha256']==source_sha
    resume=np.float64(prev['total']) if prev['total_type']=='numpy.float64' else float(prev['total']);previous_sha=hashlib.sha256(previous_path.read_bytes()).hexdigest()
   descriptor=prep['cells'][k];poly=fam['polys'][descriptor['poly']];cell=[np.array(v,dtype=np.float64) for v in descriptor['vertices']]
   local_fam=dict(fam,polys=[poly]);calls=[]
   def captured_cells(received_model,received_poly,B,received_ops):
    assert received_model is model and received_poly is poly and B is fam['B'] and received_ops is ops
    calls.append(k);return [cell]
   FX.cells_on_face=captured_cells
   tree=ast.parse((T.SRC/'fibre4b.py').read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='face_err'];assert len(tree.body)==1
   replacements=0
   for node in tree.body[0].body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='tot':
     assert isinstance(node.value,ast.Constant) and node.value.value==0.0;node.value=ast.Name(id='_resume_total',ctx=ast.Load());replacements+=1
   assert replacements==1;F3._resume_total=resume;tree=T.Rewrite().visit(tree);exec(compile(ast.fix_missing_locations(tree),'T180 saved-running-total continuation','exec'),F3.__dict__)
   stats={'traces':0,'pieces':0};value=F3.face_err(model,local_fam,side,ops,stats);assert calls==[k]
   assert type(value) in (float,np.float64)
   out.update(cell=k,poly=descriptor['poly'],initial_total=float(resume),total=float(value),total_hex=float(value).hex(),total_type='numpy.float64' if isinstance(value,np.float64) else 'float',stats=stats,legacy_ops=ops.n,prepare_sha256=hashlib.sha256(prep_path.read_bytes()).hexdigest(),previous_sha256=previous_sha)
  out['complete']=True
 except Exception as exc:out['failure']=repr(exc);raise
 finally:
  out.update(ledger=ledger.report() if ledger else None,cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('ledger','cells')}),flush=True)
elif mode=='summarize':
 prep=json.loads((folder/'prepare.json').read_text());assert prep['complete'];rows=[json.loads((folder/f'cell_{k:05d}.json').read_text()) for k in range(prep['cell_count'])];assert rows and all(r['complete'] and r['cell']==k for k,r in enumerate(rows))
 assert all(r['source_sha256']==source_sha for r in [prep]+rows)
 prepare_sha=hashlib.sha256((folder/'prepare.json').read_bytes()).hexdigest()
 for k,r in enumerate(rows):
  assert r['prepare_sha256']==prepare_sha and r['model_sha256']==inv['model_sha256']
  assert r['initial_total'].hex()==(rows[k-1]['total'].hex() if k else 0.0.hex())
  assert r['previous_sha256']==(hashlib.sha256((folder/f'cell_{k-1:05d}.json').read_bytes()).hexdigest() if k else None)
 assert rows[-1]['total_hex']==expected['raw_error_component'].hex()
 stats={key:sum(r['stats'][key] for r in rows) for key in ('traces','pieces')};assert stats==expected['stats']
 actual_legacy=prep['legacy_ops']+sum(r['legacy_ops'] for r in rows);repeat=(len(rows)-1)*16*prep['clamps'];assert actual_legacy-repeat==expected['legacy_ops']
 sites={};totals=Counter();opaque=[]
 for row in [prep]+rows:
  l=row['ledger'];check=Counter()
  for site,counts in l['sites'].items():check.update(counts);sites.setdefault(site,Counter()).update(counts)
  assert dict(check)==l['totals'];totals.update(check);opaque.extend(l['opaque_calls'])
 numeric=sum(v for k,v in totals.items() if not k.startswith('integer_control'));integer=sum(v for k,v in totals.items() if k.startswith('integer_control'))
 out={'complete':True,'index':idx,'family':expected['family'],'side':expected['side'],'model_sha256':inv['model_sha256'],'raw_error_component':rows[-1]['total'],'stats':stats,'legacy_ops':actual_legacy-repeat,'actual_adapter_legacy_ops':actual_legacy,'repeated_legacy_setup':repeat,'cells':len(rows),'ledger':{'sites':sites,'totals':totals,'named_numeric_total':numeric,'integer_control_total':integer,'inclusive_explicit_total':numeric+integer,'opaque_calls':opaque},'cpu_seconds':sum(r['cpu_seconds'] for r in [prep]+rows),'max_rss_kib':max(r['max_rss_kib'] for r in [prep]+rows),'scope':'Exact-order separate-process cell adapter, repeated clamp numerical setup included. Original source arithmetic upper bound, not equal cost. Resource failures accounted separately.'}
 if idx==6:
  original=json.loads((HERE/'components/06.json').read_text());diff={};expected_diff={}
  for site in set(sites)|set(original['ledger']['sites']):
   a=Counter(sites.get(site,{}));b=Counter(original['ledger']['sites'].get(site,{}));delta={k:a[k]-b[k] for k in set(a)|set(b) if a[k]!=b[k]}
   if delta:diff[site]=delta
  first=rows[0]['ledger']['sites'];expected_diff={site:{k:v*(len(rows)-1) for k,v in counts.items()} for site,counts in first.items() if site.startswith('_fn:') and len(rows)>1}
  assert diff==expected_diff,(diff,expected_diff)
  out['control_site_difference']=diff
 (folder/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('ledger','control_site_difference')}))
else:raise ValueError(mode)
