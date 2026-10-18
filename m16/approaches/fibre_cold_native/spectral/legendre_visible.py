"""Explicit NumPy Legendre wrapper expressions; eigen kernel stays opaque."""
import ast,hashlib,importlib.util,json,resource,time,types
from pathlib import Path
import numpy as np
from numpy.polynomial import legendre as original
HERE=Path(__file__).resolve().parent
BASE=Path('/workspace/work/recover-small_h2_78/m16/approaches/fibre_cost_review/arithmetic.py')
assert hashlib.sha256(BASE.read_bytes()).hexdigest()=='48bf04ead5510fc301a8db9732d90e00179d0ae6524f515344f28df2a94385db'
spec=importlib.util.spec_from_file_location('reviewed_arithmetic_t180',BASE);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

class Ledger(base.Ledger):
    def __init__(self):super().__init__();self.calls=[]
    def call(self,site,name,fn,*args,**kwargs):
        result=fn(*args,**kwargs);self.calls.append({'site':site,'name':name})
        if name in ('np.sqrt','np.abs'):self.charge(site,'special_'+name,np.size(result))
        elif name in ('.max','.sum'):
            n=np.size(fn.__self__);self.charge(site,'reduce_'+name,max(n-np.size(result),0))
        elif name not in ('np.array','np.zeros','np.arange','np.empty','np.moveaxis','np.lib.array_utils.normalize_axis_index','.astype','np.linalg.eigvalsh','pu.as_series','pu._as_int'):
            raise ValueError('Unclassified helper '+name)
        return result

class Rewrite(base.Rewrite):
    def visit_Call(self,node):
        name=ast.unparse(node.func)
        if isinstance(node.func,ast.Attribute) and node.func.attr in ('max','sum','astype') and not (isinstance(node.func.value,ast.Name) and node.func.value.id in ('np','pu')):name='.'+node.func.attr
        if name.startswith(('np.','pu.')) or name in ('.max','.sum','.astype'):
            site=self.site(node);node=self.generic_visit(node);out=self.wrap(node,'call',[ast.Constant(site),ast.Constant(name),node.func]+node.args);out.keywords=node.keywords;return out
        return super().visit_Call(node)

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,22));dest=HERE/'legendre_visible.json';assert not dest.exists()
    manifest=json.loads((HERE/'capture_manifest.json').read_text())
    source=Path(original.__file__);assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['files'][str(source)]
    saved=json.loads((HERE.parent/'capture.json').read_text());assert saved['complete']
    rows=[];start=time.process_time();out={'complete':False,'rows':rows,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'explicit four-function expressions only; eigenvalues and pu.as_series/_as_int internals retained unpriced'}
    try:
        for i,row in enumerate(saved['rows']):
            if row['name']!='leggauss':continue
            ledger=Ledger();mod=types.ModuleType('isolated_legendre');mod.__dict__.update(original.__dict__);mod._arithmetic=ledger
            tree=ast.parse(source.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('leggauss','legcompanion','legval','legder')];assert len(tree.body)==4
            exec(compile(ast.fix_missing_locations(Rewrite().visit(tree)),str(source)+' [visible cold count]','exec'),mod.__dict__)
            n=row['args']['items'][0];record={'saved_row':i,'order':n};rows.append(record)
            try:
                value=mod.leggauss(n);encoded=[[float(x).hex() for x in a] for a in value];expected=[x['hex'] for x in row['output']['items']]
                record.update(output_hex=encoded,parity=encoded==expected);assert record['parity']
            finally:record.update(ledger=ledger.report(),calls=ledger.calls)
        assert len(rows)==6;out['complete']=True
    except Exception as exc:out['failure']=repr(exc);raise
    finally:
        out.update(cpu=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
if __name__=='__main__':main()
