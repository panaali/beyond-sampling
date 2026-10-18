"""Independent AST/JSON metadata audit only. Never import author modules."""
import ast, hashlib, json, math, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
AUTHOR=Path('/workspace/work/recover-small_h2_78/review/recipe112_sensitivity_recovery')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def calls(tree,attr):
 return [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr==attr]
def assignment(tree,name):
 return next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
def main():
 start=time.process_time();out=HERE/'results.json';assert not out.exists()
 d=json.loads((AUTHOR/'inventory.json').read_text())
 for p,h in d['pins'].items():assert sha(Path(p))==h,p
 paths={Path(p).name:Path(p) for p in d['pins']}
 sens=ast.parse(paths['sens.py'].read_text());rec=ast.parse(paths['rec112.py'].read_text());mine=ast.parse(paths['mine.py'].read_text())
 pclass=next(n for n in rec.body if isinstance(n,ast.ClassDef) and n.name=='P')
 defaults={t.id:ast.literal_eval(n.value) for n in pclass.body if isinstance(n,ast.Assign) for t in n.targets}
 assert defaults==d['defaults']
 variants=assignment(sens,'VARIANTS');assert isinstance(variants,ast.Dict)
 independent=[]
 for k,v in zip(variants.keys,variants.values,strict=True):
  assert isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id=='V' and not v.args
  override={q.arg:ast.literal_eval(q.value) for q in v.keywords}
  independent.append(dict(name=ast.literal_eval(k),overrides=override,effective=defaults|override))
 assert independent==d['variants'] and len(independent)==16
 assert all(x['effective']['dense']==0 and x['effective']['joint'] is False for x in independent)
 main=next(n for n in sens.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 nc=calls(main,'noisy_abs');assert len(nc)==1 and [ast.literal_eval(a) for a in nc[0].args[1:]]==[16,.05,86]
 assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='range' and len(n.args)==1 and ast.literal_eval(n.args[0])==5 for n in ast.walk(main))
 counted=calls(main,'counted');assert len(counted)==1 and [ast.unparse(a) for a in counted[0].args]==['m','pr']
 ms=[n.value for n in ast.walk(main) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ms' for t in n.targets)]
 assert len(ms)==1 and ast.unparse(ms[0])=='[b] + M.noisy_abs(b, 16, 0.05, 86)'
 num=next(n for n in rec.body if isinstance(n,ast.ClassDef) and n.name=='Num')
 methods={n.name for n in num.body if isinstance(n,ast.FunctionDef)};assert not {'__eq__','__ne__'}&methods
 countedfn=next(n for n in rec.body if isinstance(n,ast.FunctionDef) and n.name=='counted')
 assert 'rec112(as_lists(model, True), pr)' in ast.unparse(countedfn)
 noisefn=next(n for n in mine.body if isinstance(n,ast.FunctionDef) and n.name=='noisy_abs')
 assert [ast.unparse(c.args[0]) for c in calls(noisefn,'standard_normal')]==['Wi.shape','Wh.shape','Wo.shape']
 assert [ast.unparse(c.args[0]) for c in calls(noisefn,'default_rng')]==['seed']
 zoofn=next(n for n in mine.body if isinstance(n,ast.FunctionDef) and n.name=='zoo_w')
 z=calls(zoofn,'zoo_2nd_argmax');assert len(z)==1 and [k.arg for k in z[0].keywords]==['seed']
 coverage=[];names=[v['name'] for v in independent]
 for H in (4,5,6,8):
  raw=json.loads(paths[f'sens_H{H}.json'].read_text());assert list(raw)==names
  for records in raw.values():
   assert set(records)==set(map(str,range(5)))
   for r in records.values():
    assert set(r)=={'clean_err','ratio','max_abs','ops'} and all(isinstance(v,(int,float)) and math.isfinite(v) for v in r.values())
  coverage.append(dict(H=H,variant_groups=len(raw),seed_groups=sum(map(len,raw.values())),saved_fields=sorted(next(iter(next(iter(raw.values())).values()))),individual_model_rows_present=False))
 assert coverage==d['historical_summary_coverage']
 assert len(coverage)*len(independent)*5*17==d['planned_source_recipe_calls_if_entire_original_panel_replayed']==5440
 assert len(coverage)*5*17==d['unique_model_inputs_by_source_design']==340
 sem=json.loads(paths['semantics.json'].read_text());assert sem['models']==d['already_reviewed_frozen_diagnostics']
 assert [(r['H'],r['zoo'],r['noise_seed'],r['copy']) for r in sem['models']]==[(8,2,89,3),(4,1,89,2)]
 assert sem['primitives']['False'][0]['eq_zero'] is False and sem['primitives']['True'][0]['eq_zero'] is True
 assert not list(paths['sens_H4.json'].parent.glob('*86*'))
 result=dict(complete=True,claim='5c6d2cb4',pins={str(p):sha(p) for p in [AUTHOR/'inventory.json',AUTHOR/'README.md',HERE/'check_saved.py']}|d['pins'],checked_variants=16,checked_aggregate_groups=320,source_recipe_calls=5440,source_model_slots=340,source_additional_truth_calls=340,source_sampling_counter_setups=4,all_coverage=coverage,author_imports=0,model_estimator_noise_truth_mass_native_calls=0,cpu_seconds=time.process_time()-start,scope='Source design and saved metadata only; no numerical sensitivity validation, historical checkpoint proof, panel restoration or replay authorization.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('pins','all_coverage')}))
if __name__=='__main__':main()
