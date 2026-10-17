"""Saved-word operation-shape census, not a numerical cache implementation."""
from collections import Counter

def validate(words,H):
 if type(H) is not int or H<=0:raise ValueError('H must be a positive integer')
 if not words or any(not isinstance(s,str) for s in words):raise ValueError('nonempty string histories required')
 n=len(words[0])
 if n==0 or n%H or any(len(s)!=n or set(s)-{'0','1'} for s in words):raise ValueError('equal positive H-bit history lengths required')
 T=n//H
 if T<2:raise ValueError('at least two steps required')
 return sorted(set(words)),T

def dot_work(H,columns,repetitions):
 # Actual shape loops over scalar multiplication/addition SITES, no weights.
 c=Counter(mul=0,add=0)
 for _ in range(repetitions):
  for row in range(H):
   for col in range(columns):
    for k in range(H):
     c['mul']+=1
     if k:c['add']+=1
 assert c==dict(mul=repetitions*H*H*columns,add=repetitions*H*(H-1)*columns)
 return dict(c)

def add(*rows):return {k:sum(r[k] for r in rows) for k in ('mul','add')}
def total(row):return row['mul']+row['add']

def census(words,H):
 unique,T=validate(words,H);prefixes=[sorted({w[:H*t] for w in unique}) for t in range(T+1)]
 base_sites=[dot_work(H,t,len(prefixes[t])) for t in range(T)]
 baseline=add(*base_sites);cuts=[]
 for a in range(1,T):
  b=T-a;suffix=[sorted({w[H*a:H*(a+j)] for w in unique}) for j in range(b)]
  cold_sites=[dot_work(H,H+j,len(suffix[j])) if j else dict(mul=0,add=0) for j in range(b)]
  prefix=add(*base_sites[:a]);sub_sites=[dot_work(H,a,len(prefixes[t])) for t in range(a,T)]
  substitutions=add(*sub_sites);warm=add(prefix,substitutions);cold=add(*cold_sites);combined=add(warm,cold)
  c=H*(2*H-1)
  assert total(warm)==c*(sum(t*len(prefixes[t]) for t in range(a))+a*sum(len(prefixes[t]) for t in range(a,T)))
  assert total(cold)==c*sum((H+j)*len(suffix[j]) for j in range(1,b))
  cuts.append(dict(cut=a,suffix_parent_prefixes=suffix,prefix_work=prefix,substitution_sites=sub_sites,substitutions=substitutions,cold_sites=cold_sites,warm=warm,cold=cold,combined=combined,named_saving=total(baseline)-total(combined),logical_cache_preactivation_entries=sum(len(suffix[j])*H*(H+j+1) for j in range(b))))
 assert total(baseline)==H*(2*H-1)*sum(t*len(prefixes[t]) for t in range(T))
 return dict(H=H,T=T,input_rows=len(words),unique_histories=unique,duplicate_rows=len(words)-len(unique),parent_prefixes=prefixes,baseline_sites=base_sites,baseline=baseline,cuts=cuts)
