"""Exact necessary readout condition for global positive-monomial automorphisms.
Rows/entries must be list or tuple and exact int/Fraction, never binary floats.
No inference about argmax-only equivalence, reachable domains or noninvertible quotients.
"""
from fractions import Fraction as F
from itertools import combinations

def enc(q):return str(q.numerator)+'/'+str(q.denominator)
def readout_predicate(rows):
 if not isinstance(rows,(list,tuple)) or not rows:raise ValueError('nonempty row matrix required')
 if any(not isinstance(row,(list,tuple)) for row in rows):raise TypeError('rows must be list or tuple')
 h=len(rows[0])
 if not h or any(len(row)!=h for row in rows):raise ValueError('positive rectangular width required')
 if any(not (type(x) is int or isinstance(x,F)) for row in rows for x in row):raise TypeError('only exact int/Fraction coefficients')
 cols=[tuple(F(row[j]) for row in rows) for j in range(h)]
 zero=[];canonical=[];groups={};cert=[];normalization_divisions=0;minor_evaluations=0
 for j,col in enumerate(cols):
  nonzero=next((i for i,x in enumerate(col) if x),None)
  if nonzero is None:zero.append(j);canonical.append(None);continue
  scale=abs(col[nonzero]);key=tuple(x/scale for x in col);normalization_divisions+=len(col)
  canonical.append(key);groups.setdefault(key,[]).append(j)
 collisions=[v for v in groups.values() if len(v)>1]
 for i,j in combinations(range(h),2):
  if canonical[i] is None or canonical[j] is None:
   cert.append({'pair':[i,j],'kind':'zero_column'});continue
  if canonical[i]==canonical[j]:
   pivot=next(k for k,x in enumerate(cols[i]) if x);ratio=cols[j][pivot]/cols[i][pivot]
   assert ratio>0 and all(b==ratio*a for a,b in zip(cols[i],cols[j]))
   cert.append({'pair':[i,j],'kind':'positive_proportional','ratio_j_over_i':enc(ratio)});continue
  found=None
  for a,b in combinations(range(len(rows)),2):
   det=cols[i][a]*cols[j][b]-cols[i][b]*cols[j][a];minor_evaluations+=1
   if det:
    found={'pair':[i,j],'kind':'nonzero_minor','rows':[a,b],'determinant':enc(det)};break
  if found is None:
   pivot=next(k for k,x in enumerate(cols[i]) if x);ratio=cols[j][pivot]/cols[i][pivot]
   assert ratio<0 and all(b==ratio*a for a,b in zip(cols[i],cols[j]))
   found={'pair':[i,j],'kind':'negative_proportional','ratio_j_over_i':enc(ratio)}
  cert.append(found)
 return {'rows':len(rows),'width':h,'columns':[[enc(x) for x in col] for col in cols],
         'positive_canonical_columns':[None if q is None else [enc(x) for x in q] for q in canonical],
         'zero_columns':zero,'positive_proportional_classes':collisions,'pair_certificates':cert,
         'status':'identity_forced_by_readout' if not zero and not collisions else 'inconclusive',
         'partial_setup_counts':{'normalization_divisions':normalization_divisions,'minor_evaluations':minor_evaluations},
         'scope':'global-state positive-monomial unchanged-logit weight automorphisms only',
         'not_complete_native_or_bit_cost':True}
