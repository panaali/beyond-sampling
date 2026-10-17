"""No-change walk-cap discriminator for the frozen12 current models."""
import ast,json,time
from pathlib import Path
from collections import defaultdict
from loader import load
from models import frozen_models
HERE=Path(__file__).parent

class CheckedRange:
    def __init__(self,records,site,n):
        self.record={'site':site,'iterations':0,'exhausted':False};records.append(self.record);self.n=n
    def __iter__(self):return self
    def __next__(self):
        k=self.record['iterations']
        if k==self.n:self.record['exhausted']=True;raise StopIteration
        self.record['iterations']+=1;return k

class Rewrite(ast.NodeTransformer):
    def __init__(self):self.function='module';self.sites=[]
    def visit_FunctionDef(self,node):
        prev=self.function;self.function=node.name;node=self.generic_visit(node);self.function=prev;return node
    def visit_For(self,node):
        if ast.unparse(node.iter)=='range(64)':
            self.sites.append(self.function);node.iter=ast.Call(ast.Name('_checked_range',ast.Load()),[ast.Constant(self.function),ast.Constant(64)],[])
        return self.generic_visit(node)

def main():
    begin=time.process_time();prior=json.loads((HERE/'panel_results.json').read_text());byname={r['name']:r['baseline'] for r in prior['rows']};rows=[]
    for name,model,digest in frozen_models():
        mod,_,_=load();records=[];mod._checked_range=lambda site,n:CheckedRange(records,site,n);tree=ast.parse((HERE/'snapshot/fibre_fast.py').read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef)];rw=Rewrite();exec(compile(ast.fix_missing_locations(rw.visit(tree)),'fast cap audit','exec'),mod.__dict__);assert sorted(rw.sites)==['forward_trace','reverse_walk']
        ops=mod.Ops();info={};value=mod.est_fast(model,ops,info);base=byname[name];assert float(value).hex()==base['value_hex'] and ops.n==base['n'];assert all(info[k]==base[k] for k in ('traces','segments','cons'))
        summary={}
        for site in rw.sites:
            found=[r for r in records if r['site']==site];summary[site]=dict(calls=len(found),maximum_iterations=max((r['iterations'] for r in found),default=0),exhaustions=sum(r['exhausted'] for r in found))
        rows.append(dict(name=name,summary=summary,all_exhausted_records=[r for r in records if r['exhausted']]))
    out=dict(rows=rows,cpu_seconds=time.process_time()-begin,scope='only inherited64-step caps on frozen12; no universal termination or floating topology claim');(HERE/'cap_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
