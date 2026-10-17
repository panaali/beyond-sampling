"""Saved-data contraction only: never imports or executes the estimator."""
from pathlib import Path
from collections import Counter,defaultdict
import ast,hashlib,json,time,resource,traceback,math
P=Path(__file__).resolve().parent

def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def keyed(rows):
 out=Counter()
 for r in rows:out[tuple(r['key'])]+=r['count']
 return out

def validate(l):
 total=Counter()
 for counts in l['sites'].values():total.update(counts)
 assert dict(total)==l['totals']
 numeric=sum(v for k,v in total.items() if not k.startswith('integer_control'));integer=sum(v for k,v in total.items() if k.startswith('integer_control'))
 assert numeric==l['named_numeric_total'] and integer==l['integer_control_total'] and numeric+integer==l['inclusive_explicit_total']
 return total

def shape(s):
 x=ast.literal_eval(s)
 if not isinstance(x,tuple) or not all(isinstance(v,int) and v>=0 for v in x):return None
 return x

def group(name):
 if name in ('np.linalg.norm','np.dot'):return 'norm_or_dot'
 if name in ('np.min','np.max','np.sum','np.any','np.all','np.argmax','np.argmin','.min','.max','.sum','.mean','.any','.all','np.mean'):return 'reduction'
 if name in ('np.arccos','np.arctan') or name.startswith('math.'):return 'elementary_function'
 if name in ('np.abs','np.sqrt','np.isnan','np.minimum','np.maximum','np.clip','abs'):return 'elementwise_primitive'
 if name in ('np.cross','np.outer'):return 'composite_array_primitive'
 if name.startswith('np.linalg.') or name=='np.polynomial.legendre.leggauss':return 'opaque_linalg'
 return 'construction_or_unclassified'

def main():
 result=P/'results.json';assert not result.exists(),'preserve immutable attempt'
 protocol=read(P/'protocol.json');t=time.process_time();out=dict(complete=False,protocol_sha256=digest(P/'protocol.json'),source_sha256=digest(__file__),phases=[])
 def save():
  out['cpu']=time.process_time()-t;out['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;result.write_text(json.dumps(out,indent=2)+'\n')
 save()
 try:
  for path,sha in protocol['pins'].items():assert digest(path)==sha,path
  out['phases'].append('all source and data pins match');save()
  components=[read(p) for p in protocol['components']];assert len(components)==48 and all(r['complete'] and r.get('index',0)==i for i,r in enumerate(components))
  cells=[read(p) for p in protocol['rescued_cells']];prep=read(protocol['rescue_prepare']);assert prep['complete'] and len(cells)==144 and all(r['complete'] and r['cell']==i for i,r in enumerate(cells))
  rescue=components[12]['ledger'];rsites=defaultdict(Counter);rtypes=Counter();rcalls=Counter();rmem=Counter();ropaque=Counter()
  for r in [prep]+cells:
   l=r['ledger'];validate(l);rtypes.update(l['totals']);rcalls.update(keyed(l['native_calls']));rmem.update(keyed(l['memory_calls']));ropaque.update(keyed(l['opaque_calls']))
   for site,counts in l['sites'].items():rsites[site].update(counts)
  assert dict(rtypes)==rescue['totals'] and {s:dict(c) for s,c in rsites.items()}==rescue['sites'] and ropaque==keyed(rescue['opaque_calls'])
  out['rescue_evidence']=dict(summary_omits_call_and_memory_keys='native_calls' not in rescue,raw_cells=len(cells),preparation_included=True,native_shape_keys=len(rcalls),memory_shape_keys=len(rmem))
  calls=Counter();memory=Counter();totals=Counter();diagnostic=Counter();sites=defaultdict(Counter);rows=[];opaque=Counter()
  expected=read(protocol['source_summary']);assert expected['complete']
  for i,r in enumerate(components):
   l=r['ledger'];validate(l);totals.update(l['totals']);opaque.update(keyed(l['opaque_calls']))
   if i==12:calls.update(rcalls);memory.update(rmem)
   else:calls.update(keyed(l['native_calls']));memory.update(keyed(l['memory_calls']))
   diag=Counter()
   for site,c in l['sites'].items():
    sites[site].update(c)
    if "stats['traces'] += 1" in site or "stats['pieces'] += 1" in site:diag.update(c)
   diagnostic.update(diag)
   e=expected['rows'][i];assert e['index']==i and l['named_numeric_total']==e['numeric'] and l['integer_control_total']==e['integer'] and sum(diag.values())==e['diagnostic']
   rows.append(dict(index=i,numeric=e['numeric'],integer=e['integer'],diagnostic=e['diagnostic']))
  numeric=sum(v for k,v in totals.items() if not k.startswith('integer_control'));integer=sum(v for k,v in totals.items() if k.startswith('integer_control'));diag=sum(diagnostic.values())
  assert (numeric,integer,diag)==(expected['numeric'],expected['integer'],expected['diagnostic']);assert numeric+integer-diag==expected['visible_after_diagnostic']
  adjusted=Counter({k:v-diagnostic[k] for k,v in totals.items()});assert min(adjusted.values())>=0
  callrows=[];byname=defaultdict(lambda:Counter());bygroup=defaultdict(lambda:Counter())
  for (name,ins,outs),count in sorted(calls.items()):
   si,so=shape(ins),shape(outs);n=None if si is None else math.prod(si);m=None if so is None else math.prod(so);g=group(name)
   row=dict(name=name,input_shape=ins,output_shape=outs,count=count,group=g,input_elements=n,output_elements=m,layout_stride_dtype_history_saved=False);callrows.append(row)
   for c in (byname[name],bygroup[g]):
    c['calls']+=count;c['shape_keys']+=1
    if n is not None:c['input_elements']+=n*count
    if m is not None:c['output_elements']+=m*count
  failed=read(protocol['failed12']);assert not failed['complete'] and 'failure' in failed
  composition=read(protocol['composition']);assert composition['conditional_composition']==1838695310 and composition['cold_named_subset']==295652
  out.update(complete=True,components=rows,numeric_total=numeric,integer_total=integer,diagnostic_total=diag,visible_after_diagnostic=numeric+integer-diag,categories=dict(adjusted),raw_categories=dict(totals),diagnostic_categories=dict(diagnostic),native_calls=callrows,by_name={k:dict(v) for k,v in sorted(byname.items())},by_group={k:dict(v) for k,v in sorted(bygroup.items())},native_sites={k:dict(v) for k,v in sorted(sites.items())},opaque_calls=[dict(key=list(k),count=v) for k,v in sorted(opaque.items())],memory_shape_keys=len(memory),prior_hybrid_named_composition=composition['conditional_composition'],original_failed12_cpu=failed['cpu_seconds'],zero_new_model_or_mass_evaluations=True,scope='Saved expression and call census, no installed-kernel price or full-cost claim. Binary-operation sites have counts but no operand layout/shape trace; wrapped native calls have input/output shapes but no per-call strides/dtypes/values/runtime targets. Additional primitive source/layout proof is required. Original rescue repeated setup retained. Cold and outer/candidate conventions remain separate.')
  save();print(json.dumps({k:v for k,v in out.items() if k in ('complete','cpu','max_rss_kib','numeric_total','integer_total','diagnostic_total','visible_after_diagnostic','by_group','rescue_evidence')}))
 except Exception:
  out['failure']=traceback.format_exc();save();raise
if __name__=='__main__':main()
