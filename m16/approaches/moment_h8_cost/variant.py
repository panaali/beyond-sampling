"""Original reviewed T135 moments, inserted before unchanged T161 quadrature."""
from pathlib import Path
import ast,hashlib,importlib.util,inspect,json,types
P=Path(__file__).resolve().parent
PROTOCOL=json.loads((P/'protocol.json').read_text())

def load():
 for p,h in PROTOCOL['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
 path=P.parent/'positive_h8_cost/run.py';spec=importlib.util.spec_from_file_location('original_moment_h8_selector',path);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
 old,kernel,rescue,radial=M.load()
 path=P.parent/'moment_mass_budget/primitive.py';tree=ast.parse(path.read_text());removed=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.level==2];assert len(removed)==1
 tree.body=[n for n in tree.body if n not in removed];moment=types.ModuleType('original_reviewed_moments');moment.__file__=str(path)
 for name in ('Counter','C','dot','determinant','Rules'):moment.__dict__[name]=getattr(old,name)
 exec(compile(tree,str(path),'exec'),moment.__dict__)
 source=inspect.getsource(M.select);a='a,b,det=old.geometry(R,op);';assert source.count(a)==1;source=source.replace(a,'a,b,det,data=moment.full_geometry(R,op);')
 marker='  else:\n   alpha=';assert source.count(marker)==1
 insertion='''  else:
   mu,var,eq2=moment.moments(data);out['moment']=dict(mean=mu.v,variance=var.v,second=eq2.v)
   if var<=0 or mu<=0:out['moment']['failure']='Nonpositive floating variance/mean; shortcut not used'
   else:
    lower=scale/(mu*mu);a2=a*a;upper=lower+3*scale*var/(a2*a2);ml=max(lo,lower);mh=min(hi,upper)
    if mh<ml:out['moment']['failure']='Floating interval intersection reversed; shortcut not used'
    else:
     mm=(ml+mh)/2;me=(mh-ml)/2;out['moment'].update(interval=[ml.v,mh.v],bound=me.v)
     if me<=TOL:
      out.update(mode='interval',value=mm.v,bound=me.v,moment_used=True,setup_selection_counts=op.dict());return out
   alpha='''
 source=source.replace(marker,insertion);M.moment=moment;exec(compile(source,'<original_moment_selector>','exec'),M.__dict__)
 return M,old,kernel,rescue,radial,moment,source

def one(R,M,old,kernel,rescue,radial,rules):
 row=M.one(R,old,kernel,rescue,radial,rules)
 if row['center'].get('moment_used'):row['method']='moment_interval'
 return row
