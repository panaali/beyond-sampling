"""Pinned outward Decimal evaluation of the existing ideal GL10 bound."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,importlib.util,sys,json
P=Path(__file__).resolve().parent
PROTOCOL=json.loads((P/'protocol.json').read_text())
def load():
 path=Path(PROTOCOL['directed_source']);assert hashlib.sha256(path.read_bytes()).hexdigest()==PROTOCOL['directed_sha256']
 name='pinned_reviewed_directed_tail';spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 return m
class Bound:
 def __init__(self):
  m=load();self.D=m.Directed(50);self.denominator=self.D.mul(self.D.power(self.D.pi(),2),self.D.point(6));self.cold_counts=dict(self.D.counts)
 def one(self,cert):
  D=self.D;before=Counter(D.counts);a=D.rational(F(cert['a']));b=D.rational(F(cert['b']));det=D.rational(abs(F(cert['det'])))
  tail=D.tail(a.lo,b.hi,2,8);result=D.div(D.mul(det,tail),self.denominator);counts=dict(Counter(D.counts)-before)
  return dict(lower_formula_endpoint=str(result.lo),upper_bound=str(result.hi),radial_lower_used=str(a.lo),radial_upper_used=str(b.hi),counts=counts,qualifies=bool(F(result.hi)<=F(1,10**7*35098)))
