"""Independent determinant/Cramer and Kronecker normal systems, at most order4."""
from fractions import Fraction as F
from itertools import permutations

def M(a):return [[F(x) for x in r] for r in a]
def product(a,b):
    assert a and b and len(a[0])==len(b)
    return [[sum((a[i][j]*b[j][k] for j in range(len(b))),F()) for k in range(len(b[0]))] for i in range(len(a))]
def center(a):
    k=len(a[0]);return [[(k*x-sum(row,F()))/k for x in row] for row in a]
def labels(a):return [next(i for i,x in enumerate(row) if all(x>=v for v in row)) for row in a]
def determinant(a):
    n=len(a);assert 1<=n<=4 and all(len(r)==n for r in a)
    ans=F()
    for perm in permutations(range(n)):
        term=F((-1)**sum(perm[i]>perm[j] for i in range(n) for j in range(i+1,n)))
        for i in range(n):term*=a[i][perm[i]]
        ans+=term
    return ans

class Systems:
    def __init__(self):self.rows=[];self.determinants=[]
    def solve(self,a,b):
        det=determinant(a);self.determinants.append(det)
        r={'matrix':a,'rhs':b};self.rows.append(r)
        if not det:
            r.update(status='failed',error_type='ValueError',error='singular system');return None
        n=len(a);q=len(b[0]);out=[[F() for _ in range(q)] for _ in range(n)]
        for j in range(n):
            for k in range(q):
                changed=[[b[i][k] if col==j else a[i][col] for col in range(n)] for i in range(n)]
                out[j][k]=determinant(changed)/det
        assert product(a,out)==b
        r.update(status='returned',coefficients=out);return out
    def ridge(self,x,y,penalty):
        n,d=len(x),len(x[0]);assert len(y)==n
        normal=[[sum((x[i][j]*x[i][k] for i in range(n)),F())+(F(penalty[j]) if j==k else 0) for k in range(d)] for j in range(d)]
        rhs=[[sum((x[i][j]*y[i][k] for i in range(n)),F()) for k in range(len(y[0]))] for j in range(d)]
        return self.solve(normal,rhs)
    def pair(self,x,y,penalty,convention,mean,matched=True):
        n,d,k=len(x),len(x[0]),len(y[0]);ordered=convention.startswith('ordered_');average=convention.endswith('_mean')
        pairs=[[a,b] for a in range(k) for b in range(k) if (a!=b if ordered else a<b)]
        alpha=F(2,k-1) if average else F((2 if ordered else 1)*k)
        sample=F(1,n) if mean else F(1);scale=alpha*sample;pf=scale if matched else F(1)
        # Pair Laplacian = alpha*C. The gauge rows e_j-e_last give
        # gauge Gram I+11^T and target contrasts y_j-y_last, independent
        # of the producer's expanded sample-by-pair feature construction.
        g=[[sum((row[i]*row[j] for row in x),F()) for j in range(d)] for i in range(d)]
        dim=d*(k-1)
        normal=[];rhs=[]
        for u in range(dim):
            i,j=divmod(u,k-1)
            normal.append([(scale*g[i][v//(k-1)]+(pf*F(penalty[i]) if i==v//(k-1) else 0))*(1+int(j==v%(k-1))) for v in range(dim)])
            rhs.append([scale*sum((x[t][i]*(y[t][j]-y[t][-1]) for t in range(n)),F())])
        flat=self.solve(normal,rhs);assert flat is not None
        theta=[[flat[i*(k-1)+j][0] for j in range(k-1)] for i in range(d)]
        coeff=[row+[-sum(row,F())] for row in theta]
        gauge=[[F(int(j==c)-int(c==k-1)) for c in range(k)] for j in range(k-1)]
        return dict(coefficients=coeff,pairs=pairs,data_weight=sample/(len(pairs) if average else 1),alpha=alpha,sample_factor=sample,penalty_factor=pf,gauge=gauge,theta=theta)

def reconstruct(f,index):
    s=Systems();r=dict(index=index,name=f['name'],kind=f['kind'],fixture=f,status='passed',solves=s.rows)
    x=M(f['x']);query=M(f['query']) if 'query'in f else None;kind=f['kind']
    if kind in ['commute','common','routing','pair']:
        h=M(f['hidden']);readout=M(f['readout']);y=product(h,readout);r['targets']=y
    if kind=='commute':
        bh=s.ridge(x,h,f['penalty']);bl=s.ridge(x,y,f['penalty']);pred=product(query,bl)
        r.update(hidden_coefficients=bh,logit_coefficients=bl,predictions=pred,labels=labels(pred))
        assert bh==M(f['expected_hidden']) and bl==M(f['expected_logits'])==product(bh,readout) and pred==M(f['expected_prediction']) and labels(pred)==f['expected_labels']
    elif kind=='common':
        yp=[[v+F(f['shift'][i]) for v in row] for i,row in enumerate(y)]
        b=s.ridge(x,y,f['penalty']);bp=s.ridge(x,yp,f['penalty']);diff=[[u-v for u,v in zip(a,z)] for a,z in zip(bp,b)]
        p,p2=product(query,b),product(query,bp)
        r.update(shifted_targets=yp,base_coefficients=b,shifted_coefficients=bp,coefficient_difference=diff,base_predictions=p,shifted_predictions=p2)
        assert diff==[[F(v)]*len(y[0]) for v in f['expected_shift']] and center(p)==center(p2) and labels(p)==labels(p2)==f['expected_labels']
    elif kind=='intercept':
        b=s.ridge(x,M(f['y']),f['penalty']);pred=product(query,b);r.update(coefficients=b,predictions=pred,labels=labels(pred))
        assert b==M(f['expected_coefficients']) and pred==M(f['expected_prediction']) and labels(pred)==f['expected_labels']
    elif kind=='routing':
        fits=[]
        for ids in f['groups']:
            bh=s.ridge([x[i] for i in ids],[h[i] for i in ids],f['penalty']);bl=s.ridge([x[i] for i in ids],[y[i] for i in ids],f['penalty']);assert product(bh,readout)==bl;fits.append(dict(hidden=bh,logits=bl))
        pred=[product(M([z]),fits[route]['logits'])[0] for route,z in zip(f['routes'],f['route_queries'])];r.update(fits=fits,predictions=pred);assert pred==M(f['expected_routed'])
    elif kind=='pair':
        b=s.ridge(x,y,f['penalty']);pair=s.pair(x,y,f['penalty'],f['convention'],f['sample_mean']);coef=pair['coefficients'];pred=product(query,coef)
        assert b==M(f['expected_logits']) and coef==center(b)==M([['2/3','1/6','-5/6'],['-1/3','1/3',0]]) and labels(pred)==f['expected_labels']
        resid=[[a-b for a,b in zip(row,target)] for row,target in zip(product(x,coef),y)]
        # Pair identity provides independent raw objective; also explicit
        # pair differences cross-check the fixed convention list.
        unordered=sum((len(row)*sum((v*v for v in row),F())-sum(row,F())**2 for row in resid),F())
        raw=unordered*(2 if f['convention'].startswith('ordered_') else 1)
        assert raw==sum(((row[a]-row[b])**2 for row in resid for a,b in pair['pairs']),F())
        norm=sum((v*v for row in center(resid) for v in row),F());penalty=sum((F(f['penalty'][i])*v*v for i,row in enumerate(coef) for v in row),F());data=pair['data_weight']*raw
        objective=dict(residual=resid,raw_pair_sum=raw,weighted_data=data,centered_norm=norm,unscaled_penalty=penalty,total=data+pair['penalty_factor']*penalty)
        assert data==pair['alpha']*pair['sample_factor']*norm
        r.update(ordinary=b,pair=pair,predictions=pred,objective=objective)
    elif kind=='unmatched':
        pm=s.pair(x,M(f['y']),f['penalty'],'unordered_sum',False);pu=s.pair(x,M(f['y']),f['penalty'],'unordered_sum',False,False)
        pred=[product(query,q['coefficients'])[0] for q in [pm,pu]];r.update(matched=pm,unmatched=pu,predictions=pred,labels=labels(pred))
        assert pm['coefficients']==M(f['expected_matched']) and pu['coefficients']==M(f['expected_unmatched']) and pred==M(f['expected_predictions']) and labels(pred)==f['expected_labels']
    elif kind=='specific':
        y=M(f['y']);coeff=[s.ridge(x,[[row[j]] for row in y],[pen]) for j,pen in enumerate(f['penalties'])];direct=s.ridge(x,[[sum(row,F())] for row in y],[f['sum_penalty']]);vals=[product(query,b)[0][0] for b in coeff]+[product(query,direct)[0][0]]
        r.update(coefficients=coeff,direct_sum=direct,values=vals);assert vals==[F(v) for v in f['expected']] and vals[0]+vals[1]!=vals[2]
    elif kind=='singular':
        assert s.ridge(x,M(f['y']),f['penalty']) is None
        maps=[M(f[k]) for k in ['fixed_coefficients','alternative_coefficients','sum_coefficients']];training=[product(x,b) for b in maps];queries=[product(query,b)[0] for b in maps]
        r.update(caught='singular system',training_predictions=training,query_predictions=queries)
        assert training==[M(f['y']),M(f['y']),M([[5]])] and queries==M(f['expected_queries']) and sum(queries[0],F())!=queries[2][0]
    elif kind=='relu':
        b=s.ridge(x,M(f['hidden']),f['penalty']);hidden=product(query,b);linear=product(hidden,M(f['readout']));clip=product([[max(F(),v) for v in row] for row in hidden],M(f['readout']));labs=[labels(linear)[0],labels(clip)[0]]
        r.update(coefficients=b,linear=linear,clipped=clip,labels=labs);assert linear==M(f['expected_linear']) and clip==M(f['expected_clipped']) and labs==f['expected_labels']
    else:raise AssertionError(kind)
    return r,s.determinants
