from collections import Counter,defaultdict
from fractions import Fraction
import itertools,json
from pathlib import Path
rows=[]
for T in range(3,8):
 groups=defaultdict(Counter)
 for p in itertools.permutations(range(T)):
  target=p.index(T-2);groups[p[-1]][target]+=1
 total=sum(sum(c.values()) for c in groups.values());best=sum(max(c.values()) for c in groups.values());ratio=Fraction(best,total);assert ratio==Fraction(2,T)
 for last,c in groups.items():
  if last==T-2:assert set(c)=={T-1}
  else:assert len(c)==T-1 and len(set(c.values()))==1
 rows.append(dict(T=T,permutations=total,optimal_correct=best,ceiling=str(ratio)))
Path(__file__).with_name('rank_symmetry_controls.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
