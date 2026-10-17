"""Separate inherited current helper counters from the original T95 counter."""
import ast,json,time,hashlib
from pathlib import Path
from loader import load
from models import frozen_models
HERE=Path(__file__).parent

class StripX(ast.NodeTransformer):
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='X':return None
        return self.generic_visit(node)

def source_checks():
    cur=ast.parse((HERE/'snapshot/fibre.py').read_text());old=ast.parse((HERE/'snapshot/fibre_authorcount.py').read_text());checks={}
    for name in ('rows_at','_roots_vec','Cons','trace_arc_rows'):
        a=next(n for n in cur.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name);b=next(n for n in old.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
        checks[name]=ast.dump(StripX().visit(a),include_attributes=False)==ast.dump(b,include_attributes=False)
    assert all(checks.values()),checks
    return checks

def main():
    start=time.process_time();checks=source_checks();prior=json.loads((HERE/'panel_results.json').read_text());byname={r['name']:r for r in prior['rows']};rows=[]
    for name,model,digest in frozen_models():
        mod,mods,_=load();charges=[0];oldX=mods['fibre'].X
        def counted_X(ops,k):charges[0]+=k;oldX(ops,k)
        mods['fibre'].X=counted_X;ops=mod.Ops();info={};value=mod.est_fast(model,ops,info);p=byname[name];base=p['baseline']
        assert float(value).hex()==base['value_hex'] and ops.n==base['n'];assert all(info[k]==base[k] for k in ('traces','cons','segments'))
        oldn=ops.n-charges[0];assert oldn>0
        if name=='k30_s0_clean':assert oldn==211218
        rows.append(dict(name=name,current_n=ops.n,current_helper_X=charges[0],original_author_n=oldn,expression_total=p['expression_total'],expression_sample_equivalents=p['expression_sample_equivalents'],original_author_sample_equivalents=oldn/72,ratio_to_original_author=(p['expression_total']/112)/(oldn/72),ratio_to_current_n=p['budget_ratio_vs_reported_n']))
    result=dict(helper_ast_equivalence=checks,rows=rows,cpu_seconds=time.process_time()-start,interpretation='current strict-helper Ops.n and original author counter are separate; native arithmetic unchanged')
    (HERE/'counter_provenance_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
