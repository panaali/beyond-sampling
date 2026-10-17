"""Independent saved-site totals, counter provenance and conditional comparator."""
from pathlib import Path
from fractions import Fraction as F
import ast,json,math,time
start=time.process_time();p=Path('/workspace/work/recover-small_h2_78/m16/approaches/fibre_fast_review');panel=json.loads((p/'panel_results.json').read_text());prov=json.loads((p/'counter_provenance_results.json').read_text());refs=json.loads((p/'reference_results.json').read_text());pr={r['name']:r for r in prov['rows']};rf={r['name']:r for r in refs['rows']}
class Strip(ast.NodeTransformer):
 def visit_Expr(self,n):
  if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='X':return None
  return self.generic_visit(n)
cur=ast.parse((p/'snapshot/fibre.py').read_text());old=ast.parse((p/'snapshot/fibre_authorcount.py').read_text());names=('rows_at','_roots_vec','Cons','trace_arc_rows')
for name in names:
 a=next(n for n in cur.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name);b=next(n for n in old.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name);assert ast.dump(Strip().visit(a))==ast.dump(b)
rows=[]
for r in panel['rows']:
 led=r['audit']['ledger'];sums={}
 for v in led['sites'].values():
  for k,n in v.items():sums[k]=sums.get(k,0)+n
 assert sums==led['totals'];total=sum(sums.values());assert total==r['expression_total'] and all(r['parity'].values());q=pr[r['name']];assert q['current_n']==q['original_author_n']+q['current_helper_X']==r['baseline']['n'];ratio=F(total*72,q['original_author_n']*112);assert abs(float(ratio)-q['ratio_to_original_author'])<=2*math.ulp(float(ratio))
 ref=rf[r['name']];truth=F(float(ref['truth']));estimate=F(float(ref['estimate']));assert float(estimate)==r['baseline']['value'];bias=(estimate-truth)**2*F(total,112)/(truth*(1-truth));assert abs(float(bias)-ref['conditional_squared_bias_ratio'])<=3*math.ulp(float(bias))
 rows.append({'name':r['name'],'source_expression_total':total,'historical_author_count':q['original_author_n'],'ratio_to_historical_author':float(ratio),'conditional_squared_bias_ratio':float(bias)})
out={'helper_ASTs_equivalent_except_X':list(names),'rows':rows,'cpu_seconds':time.process_time()-start,'scope':'fixed12, matched explicit-expression convention; no full protocol/hardware count or numerical truth enclosure'}
Path(__file__).with_name('fast_inventory_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
