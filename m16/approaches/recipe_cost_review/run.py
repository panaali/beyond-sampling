"""Pinned recipe-mode adapter over the independently reviewed T146 tracer."""
import ast,hashlib,json,sys,time,types,resource
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;PRIOR=HERE.parent/'plackett_cost_review'
SOURCE=Path('/workspace/work/recover-small_h2_910/m16/approaches/recipe_mass_separation')
PINS={PRIOR/'inventory.py':'25fbf143de8d93e6a9b42e0d38754b0ea8369b07af14cb65771e6586ceb800c7',
 PRIOR/'geometry.py':'9158a11b52fc01ee56b049788f9f72c05b9affbe453d77c63cbae67bacabb941',
 PRIOR/'controls.json':'4b1a28c73670311d5c38116bbc8baf5df789f1b1bc09474e7cb98d35e4a5fb94',
 SOURCE/'error_capture.json':'a873feefbcbefefa112fa25d1ade904f96f3afab67442001d5dcad5c7a068795',
 SOURCE/'error_captured.npz':'1a4ee131e9ab252a46828a1d1d2985b9d397ec1b3a626a71ddb3c00535682c44',
 SOURCE/'summary.json':'3f83f04583d2144cd02fbe53310dddf8eb4decd78edc28cbf9f852900328902e'}
for p,sha in PINS.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,str(p)
sys.path.insert(0,str(PRIOR));from inventory import Ledger,load
import inventory as inherited
class RecipeRewrite(inherited.Rewrite):
    def visit_Call(self,node):
        # A computed ndarray receiver such as np.abs(mm).max is a method,
        # unlike the namespace function np.max. Preserve its original receiver.
        if isinstance(node.func,ast.Attribute) and node.func.attr in ('min','max','sum','mean','any','all','copy','astype') and not isinstance(node.func.value,ast.Name):
            site=self.site(node);name='.'+node.func.attr;node=self.generic_visit(node)
            out=self.wrap(node,'call',[ast.Constant(site),ast.Constant(name),node.func]+node.args);out.keywords=node.keywords;return out
        return super().visit_Call(node)
inherited.Rewrite=RecipeRewrite
source=(PRIOR/'geometry.py').read_text();assert source.count("mode='exact'")==1
source=source.replace("mode='exact'","mode='recipe108'")
G=types.ModuleType('reviewed_recipe_adapter');G.__file__=str(PRIOR/'geometry.py');exec(compile(source,G.__file__+' [one mode replacement]','exec'),G.__dict__)


def controls():
    start=time.process_time();rows=[]
    models=[('zero',(np.zeros(2),np.zeros((2,2)),np.zeros((4,2)))),
            ('rational_cut',(np.array([1.,-1.]),np.array([[0.,1.],[1.,0.]]),np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])))]
    for name,model in models:
        baseline,rays=G.execute(model,0,96,False);traced,_=G.execute(model,0,96,True,np.asarray(rays))
        for key in ('legacy_geometry','stats','mass_callbacks'):assert baseline[key]==traced[key],(name,key)
        sites=traced['native']['sites']
        if name=='rational_cut':
            assert any(k.startswith('_weights_to_margin:') for k in sites)
            assert any(k.startswith('_frozen_margin:') for k in sites)
        rows.append(dict(name=name,baseline=baseline,traced=traced))
    result=dict(rows=rows,pins={str(k):v for k,v in PINS.items()},cpu_seconds=time.process_time()-start)
    (HERE/'controls.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(cpu=result['cpu_seconds'],cases=[dict(name=r['name'],stats=r['traced']['stats'],legacy=r['traced']['legacy_geometry']) for r in rows])))


def batch(index):
    start=time.process_time();assert 0<=index<8
    import torch
    torch.set_num_threads(1);s=torch.load('/workspace/work/t45_weights/2nd_argmax_8_4_1073741824_0.pth',map_location='cpu',weights_only=True)
    model=(s['rnn.weight_ih_l0'].numpy().astype(float)[:,0],s['rnn.weight_hh_l0'].numpy().astype(float),s['linear.weight'].numpy().astype(float))
    archive=np.load(SOURCE/'error_captured.npz');p=archive['parent_ids'];selected=(p>=12*index)&(p<12*(index+1))
    result,_=G.execute(model,12*index,12*(index+1),True,archive['rays'][selected])
    result.update(batch=index,cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,pins={str(k):v for k,v in PINS.items()},adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/f'geometry_{index}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('native','cold_setup','rule_setup','pins')}))

if __name__=='__main__':
    phase=sys.argv[1]
    try:controls() if phase=='controls' else batch(int(phase))
    except Exception as exc:
        (HERE/f'{phase}_failure.json').write_text(json.dumps(dict(failure=repr(exc)),indent=2)+'\n');raise
