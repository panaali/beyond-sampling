"""Exact, equally weighted finite-table diagnostics; no population estimator."""
from fractions import Fraction as F
from collections import Counter,defaultdict

def validate(rows):
    if not isinstance(rows,list) or not rows:raise ValueError('nonempty list required')
    ids=set()
    for r in rows:
        if not isinstance(r,dict) or set(r)!={'kind','index','value','label','status'}:raise ValueError('row schema')
        if r['kind'] not in ('12','23') or type(r['index']) is not int or r['index']<0:raise ValueError('identity')
        ident=(r['kind'],r['index'])
        if ident in ids:raise ValueError('duplicate identity')
        ids.add(ident)
        if type(r['value']) not in (int,F) or r['value']<0:raise ValueError('finite nonnegative exact value required')
        if not isinstance(r['label'],str) or not r['label']:raise ValueError('label required')
        if r['status']!='complete':raise ValueError('unknown status retained, no success')
    return rows

def score(A,B):
    validate(A);validate(B)
    groups=defaultdict(list)
    for r in A:groups[(r['kind'],r['label'])].append(F(r['value']))
    means={k:sum(v,F(0))/len(v) for k,v in groups.items()}
    pooled=sum((F(r['value']) for r in A),F(0))/len(A)
    residual=[];base=[];unseen=[];predictions=[]
    for r in B:
        key=(r['kind'],r['label']);pred=means.get(key,pooled)
        if key not in means:unseen.append((r['kind'],r['index']))
        predictions.append(pred);residual.append(F(r['value'])-pred);base.append(F(r['value'])-pooled)
    mse=sum((x*x for x in residual),F(0))/len(B)
    baseline=sum((x*x for x in base),F(0))/len(B)
    mean=sum(residual,F(0))/len(B)
    return dict(train_mean=pooled,means=means,predictions=predictions,residuals=residual,
                residual_mean=mean,mse=mse,baseline_mse=baseline,centered_residual_variance=mse-mean*mean,
                ratio=None if baseline==0 else mse/baseline,unseen=unseen)

def tail(rows,fraction):
    validate(rows)
    if type(fraction) not in (int,F) or not 0<fraction<=1:raise ValueError('tail fraction')
    q=F(fraction)*len(rows);n=(q.numerator+q.denominator-1)//q.denominator
    ordered=sorted(rows,key=lambda r:(-F(r['value'])**2,r['kind'],r['index']))
    return [(r['kind'],r['index']) for r in ordered[:n]]

def smaller_half(rows):
    validate(rows);counts=Counter(r['kind'] for r in rows)
    for kind,n in counts.items():
        if n%2 or {r['index'] for r in rows if r['kind']==kind}!=set(range(n)):raise ValueError('even contiguous kind indices required')
    return [dict(r) for r in rows if r['index']<counts[r['kind']]//2]

def permute_labels(rows,permutation):
    validate(rows)
    if not isinstance(permutation,list) or any(type(i) is not int for i in permutation) or sorted(permutation)!=list(range(len(rows))):raise ValueError('bijection required')
    if any(rows[i]['kind']!=rows[j]['kind'] for i,j in enumerate(permutation)):raise ValueError('kind must be preserved')
    return [dict(r,label=rows[j]['label']) for r,j in zip(rows,permutation,strict=True)]
