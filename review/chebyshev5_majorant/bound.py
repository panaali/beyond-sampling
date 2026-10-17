"""Directed fractional5D tail majorant; uses only saved exact ranges."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,importlib.util,json
P=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def protocol():
    p=json.loads((P/'protocol.json').read_text())
    for name,h in p['pins'].items():assert sha(name)==h,name
    return p

def load(p):
    spec=importlib.util.spec_from_file_location('majorant_directed',p['directed']);m=importlib.util.module_from_spec(spec)
    # dataclass resolves its defining module through sys.modules.
    import sys
    sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

class Majorant:
    def __init__(self,p):
        self.module=load(p);self.D=self.module.Directed(p['precision']);D=self.D
        self.den=D.mul(D.point(32),D.power(D.pi(),2));self.cold_counts=dict(D.counts)
    def one(self,cert,n):
        assert isinstance(n,int) and n>=2
        D=self.D;before=Counter(D.counts)
        aa=D.rational(F(cert['a']));bb=D.rational(F(cert['b']));a=D.point(aa.lo);b=D.point(bb.hi)
        assert 0<a.lo<=b.lo
        tail3=D.tail(a.lo,b.lo,3,n-2)
        scale=D.div(D.add(D.sqrt(a),D.sqrt(b)),D.point(2))
        tail5=D.mul(scale,tail3)
        det=D.rational(abs(F(cert['det'])))
        result=D.div(D.mul(det,tail5),self.den)
        return dict(order=n,degree=n-2,a_expanded=str(a.lo),b_expanded=str(b.hi),
            sqrtK_interval=[str(scale.lo),str(scale.hi)],integer3_tail=[str(tail3.lo),str(tail3.hi)],
            majorant_tail=[str(tail5.lo),str(tail5.hi)],lower_formula_endpoint=str(result.lo),
            upper_bound=str(result.hi),qualifies=F(result.hi)<=F('1e-10'),counts=dict(Counter(D.counts)-before))
