"""Independent exact decimal ratios and saved nine-case binomial qualifications."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal
from collections import Counter
import json,hashlib,ast,copy,time,resource
from scipy.special import betaincinv,betainc,betaincc
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/e46_e48_review');S=P/'snapshot';E=S/'m16/approaches/estT_fix';O=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text());rd=lambda p:json.loads(Path(p).read_text(),parse_float=Decimal);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(r,v):assert F(r['fraction'])==v and r['float']==float(v)
def risk(e,p,n):return n*(e-p)**2/(p*(1-p))
def bounds(e,lo,hi,n):
 ends=[risk(e,p,n) for p in (lo,hi)];return [F(0) if lo<=e<=hi else min(ends),max(ends)]
outpath=O/'e46_e48_saved_review_results.json';assert not outpath.exists();d=read(P/'results.json');g=read(P/'protocol.json');manifest=read(P/'manifest.json');assert d['complete'] and d['source_manifest_sha256']==sha(P/'manifest.json')==g['manifest_sha256']
for r in manifest['files']:assert sha(r['path'])==r['sha256']
assert sha(P/'audit.py')==g['audit_sha256']
old=ast.parse((S/'evidence/source_latest.py').read_text());new=ast.parse((E/'estT.py').read_text());dump=lambda n:ast.dump(n,include_attributes=False)
func=lambda t:{n.name:n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};a,b=func(old),func(new)
for k,v in a.items():
 n=copy.deepcopy(b[k])
 if k=='orthant':
  remove=[i for i,x in enumerate(n.body) if isinstance(x,ast.If) and ast.unparse(x.test)=='d == 5'];assert len(remove)==1 and ast.unparse(n.body[remove[0]])=='if d == 5:\n    return _orth5(Rays, ops)';n.body.pop(remove[0])
 assert dump(v)==dump(n)
assign=lambda t:[dump(n) for n in t.body if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign))];assert assign(new)[:len(assign(old))]==assign(old)
summary=rd(E/'summary.json');mc=rd(E/'mc_H8_T4.json');names=sorted([f'zoo{i}' for i in range(5)]+[f'zoo0_noisy{i}' for i in range(4)]);assert [r['name'] for r in d['rows']]==names and set(mc)==set(names)
progress=[json.loads(x) for x in (P/'progress.jsonl').read_text().splitlines()];assert len(progress)==19 and [x['row'] for x in progress if x['stage']=='result']==d['rows']
rows=[];totals={str(t):[0,0,0] for t in (1,2,3)};maxres=0.
for row in d['rows']:
 name=row['name'];raw=rd(E/'rescore'/f'{name}.json');e=F(raw['recipe108']['acc']);p=F(raw['exact']['acc']);cost=raw['recipe108']['ops'];n=F(cost)*F(3,2000)
 assert cost==row['legacy_cost'] and F(summary[name]['exact'])==p and F(summary[name]['est'])==e and summary[name]['ops']==cost
 for k,v in [('estimate',e),('same_source_reference',p),('legacy_sample_cost',F(2000,3)),('fractional_draws',n),('nominal_ratio',risk(e,p,n)),('floor_ratio',risk(e,p,n.numerator//n.denominator))]:check(row[k],v)
 assert abs(float(risk(e,p,n))-float(summary[name]['ratio']))<=1e-12*(1+float(summary[name]['ratio']))
 k=round(F(mc[name])*10000000);N=10000000;assert abs(F(mc[name])-F(k,N))<F(1,10**15) and row['reconstructed_MC_hits']==k and row['source_documented_MC_N']==N and not row['MC_raw_hit_ledger_saved']
 maxima={}
 for label,alpha in [('pointwise95',F(1,20)),('nine_case_bonferroni95',F(1,180))]:
  interval=row['MC_intervals'][label];lo,hi=interval['probability'];tail=float(alpha/2);ulo=float(betaincinv(k,N-k+1,tail));uhi=float(betaincinv(k+1,N-k,1-tail));assert abs(lo-ulo)<1e-12 and abs(hi-uhi)<1e-12
  residual=max(abs(float(betainc(k,N-k+1,lo))-tail),abs(float(betaincc(k+1,N-k,hi))-tail));assert residual<1e-7;maxres=max(maxres,residual)
  for field,draws in [('legacy_ratio',n),('floor_draw_ratio',F(n.numerator//n.denominator))]:
   v=bounds(e,F(lo),F(hi),draws)
   for r,x in zip(interval[field],v,strict=True):check(r,x)
  maxima[label]=interval['legacy_ratio'][1]['float']
 B=p*(1-p)/(e-p)**2
 for field,value in [('conditional_break_even_draws',B),('conditional_break_even_cost_legacy',B*F(2000,3)),('hypothetical_break_even_cost_sample512',B*512)]:check(row[field],value)
 assert not row['historical_binding']['saved_model_weight_hash'] and not row['historical_binding']['saved_source_hash']
 for t,v in raw['splits'].items():
  for j,key in enumerate(('n','class_changes','err_changes')):totals[t][j]+=v[key]
 rows.append(dict(name=name,nominal=float(risk(e,p,n)),upper=maxima))
assert totals==d['diagnostic_totals']
for r in d['conditional_positive_reference_comparisons']:
 orig=next(x for x in d['rows'] if x['name']==r['name']);p=F(r['positive_reference']['fraction']);eps=F(r['conditional_bound']['fraction']);e=F(orig['estimate']['fraction']);n=F(orig['fractional_draws']['fraction'])
 source=rd(S/'evidence'/('positive_seed0.json' if r['name']=='zoo0' else 'positive_seed1.json'))
 assert p==F(source['positive_accuracy'] if r['name']=='zoo0' else source['exact']['accuracy_positive']) and eps==F(source['conditional_integration_bound'] if r['name']=='zoo0' else source['exact']['conditional_bound'])
 check(r['published_reference_minus_positive'],F(orig['same_source_reference']['fraction'])-p)
 for encoded,v in zip(r['legacy_ratio_range'],bounds(e,p-eps,p+eps,n),strict=True):check(encoded,v)
assert d['cost_evidence']['partial_hybrid']==3591731780 and not d['cost_evidence']['cost_multiplier_transfer_allowed'] and not d['coverage']['full_rung_confirmed'] and d['model_mass_MC_calls']==0
prior=read(S/'evidence/t5_evidence.json');assert d['reused_T5_review']['review']=='199420a1' and d['reused_T5_review']['nominal_ratio']==prior['one_model']['nominal_ratio']['float']
out={'pass':True,'cases':9,'rows':rows,'max_beta_tail_residual':maxres,'positive_references_conditional':2,'result_sha256':sha(P/'results.json'),'manifest_sha256':sha(P/'manifest.json'),'review_cpu':time.process_time()-start,'scope':'Saved decimal ratios, nine conditional source-documented binomial boxes and source parity only. Historical hashes/raw MC ledger absent; positive references conditional, hybrid exact seed0 mass cost cannot transfer to recipe/family. No new model/reference/mass calls or full-rung claim.'}
outpath.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
