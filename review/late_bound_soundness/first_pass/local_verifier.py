"""T463 independent local primitives. Exact real semantics for supplied binary floats.
No model loading, search, or promotion of an upstream floating CROWN bound.
"""
from fractions import Fraction as F
import math


def rational(v):
    if isinstance(v, F):
        return v
    if not math.isfinite(float(v)):
        raise ValueError('nonfinite input')
    return F(v)


def _box(lo, hi):
    if len(lo) != len(hi) or not lo:
        raise ValueError('box shape')
    l, h = list(map(rational, lo)), list(map(rational, hi))
    if any(a > b for a, b in zip(l, h)):
        raise ValueError('inverted box')
    return l, h


def exact_crown(lo, hi, wi, wh, wo, A, c, target=0):
    """Return exact lower bounds of logits[j]-logits[target] for a homogeneous
    RNN h[t]=ReLU(wh*h[t-1]+wi*(A[t]*u+c[t])), h[-1]=0.
    Original wo rows are subtracted EXACTLY. All float inputs mean exact dyadics.
    A proposed external bound is never trusted. A caller prunes only if >0.
    """
    lo, hi = _box(lo, hi)
    wi = list(map(rational, wi)); H = len(wi); D = len(lo)
    wh = [list(map(rational, row)) for row in wh]
    wo = [list(map(rational, row)) for row in wo]
    A = [list(map(rational, row)) for row in A]; c = list(map(rational, c))
    if (H == 0 or len(wh) != H or any(len(r) != H for r in wh+wo)
            or not wo or not 0 <= target < len(wo) or len(A) != len(c)
            or not A or any(len(r) != D for r in A)):
        raise ValueError('network shape')
    def lin(row, low, high):
        return (sum(a*(l if a >= 0 else h) for a,l,h in zip(row,low,high)),
                sum(a*(h if a >= 0 else l) for a,l,h in zip(row,low,high)))
    low = [F(0)]*H; high = low[:]; pre = []
    for row, bias in zip(A,c):
        yl,yh = lin(row,lo,hi); yl += bias; yh += bias
        zl=[]; zh=[]
        for i in range(H):
            l,h = lin(wh[i],low,high)
            zl.append(l+wi[i]*(yl if wi[i]>=0 else yh))
            zh.append(h+wi[i]*(yh if wi[i]>=0 else yl))
        pre.append((zl,zh)); low=[max(F(0),v) for v in zl]; high=[max(F(0),v) for v in zh]
    bounds=[]
    for out in range(len(wo)):
        if out == target:
            continue
        q=[wo[out][i]-wo[target][i] for i in range(H)]
        p=[F(0)]*D; b=F(0)
        for t in reversed(range(len(A))):
            v=[]
            for qi,l,h in zip(q,*pre[t]):
                if l >= 0: slope,intercept=F(1),F(0)
                elif h <= 0: slope,intercept=F(0),F(0)
                elif qi < 0: slope=h/(h-l); intercept=-l*slope
                else: slope,intercept=F(h >= -l),F(0)
                b += qi*intercept; v.append(qi*slope)
            scale=sum(v[i]*wi[i] for i in range(H))
            p=[p[k]+scale*A[t][k] for k in range(D)]; b+=scale*c[t]
            q=[sum(v[i]*wh[i][j] for i in range(H)) for j in range(H)]
        bounds.append(b+lin(p,lo,hi)[0])
    return bounds


def _arb_q(q):
    from flint import arb
    q=rational(q)
    return arb(q.numerator)/arb(q.denominator)


def mass_upper(lo,hi,nr=4,ng=8,prec=128):
    """Upper float for true Gaussian slab mass, not only a float expression.
    Valid finite 9-D box with nonnegative a/l coordinates. g>=5/2, tail g>20.
    Entire mathematical expression enclosed by Arb; upward final conversion.
    Tail uses analytical universal min||y||^2>=1/2. No caller-supplied norm bound.
    """
    from flint import arb, ctx
    lo,hi=_box(lo,hi)
    if len(lo)!=9 or any(v<0 for v in lo[1:]): raise ValueError('domain')
    if not isinstance(nr,int) or not isinstance(ng,int) or min(nr,ng)<1 or prec<64:
        raise ValueError('grid/precision')
    if any(l==h for l,h in zip(lo,hi)): return 0.0,arb(0)
    with ctx.workprec(prec):
        aa=_arb_q; pi2=2*arb.pi(); sq2=arb(2).sqrt()
        def amin(vals):
            ans=vals[0]
            for v in vals[1:]: ans=ans.min(v)
            return ans
        def amax(vals):
            ans=vals[0]
            for v in vals[1:]: ans=ans.max(v)
            return ans
        def ds(cl,ch,rl,rh,gl,gh):
            zl=amin([g*(r+cl) for g in (gl,gh) for r in (rl,rh)])
            zh=amax([g*(r+ch) for g in (gl,gh) for r in (rl,rh)])
            # For ambiguous sign, phi(0) is a valid bound.
            d=zl if zl>0 else -zh if zh<0 else arb(0)
            b2=gh*(ch-cl)*(-d*d/2).exp()/pi2.sqrt()
            b1=((zl/sq2).erfc()-(zh/sq2).erfc())/2
            return b1.min(b2).max(arb(0))
        rr=[lo[0]+(hi[0]-lo[0])*F(i,nr) for i in range(nr+1)]
        gg=[F(5,2)+F(35,2)*F(i,ng)**2 for i in range(ng+1)]
        total=arb(0)
        for i in range(nr):
            rl,rh=aa(rr[i]),aa(rr[i+1]); rc=aa(min(max(F(-1,2),rr[i]),rr[i+1]))
            q=2*(rc+aa(F(1,2)))**2+aa(F(1,2))
            for j in range(ng):
                gl,gh=aa(gg[j]),aa(gg[j+1]); f=gh/pi2*(-gl*gl*q/2).exp()
                f*=ds(1+aa(lo[1]),1+aa(hi[1]),rl,rh,gl,gh)
                for k in range(2,9): f*=ds(-aa(hi[k]),-aa(lo[k]),rl,rh,gl,gh)
                total+=f*(rh-rl)*(gh-gl)
        vol=arb(1)
        for l,h in zip(lo,hi): vol*=aa(h-l)
        w=arb(100)
        total+=vol*16/pi2**5*32*24*(-w).exp()*sum(w**k/math.factorial(k) for k in range(5))
        endpoint=float(total.upper())
        if not math.isfinite(endpoint) or endpoint<0: raise ArithmeticError('unresolved enclosure')
        return math.nextafter(endpoint,math.inf),total


def positive_subregion_lower(lo,hi,prec=256):
    """A true mass lower bound on whole supplied r interval x g in [2.5,3].
    Density infimum times widths; intentionally crude but strictly positive.
    """
    from flint import arb,ctx
    lo,hi=_box(lo,hi)
    with ctx.workprec(prec):
        aa=_arb_q; pi2=2*arb.pi(); gl=F(5,2); gh=F(3)
        r0,r1=lo[0],hi[0]
        qmax=max(r0*r0+(r0+1)**2,r1*r1+(r1+1)**2)
        v=aa(gl)/pi2*(-aa(gh*gh*qmax)/2).exp()
        for k in range(1,9):
            cl,ch=(1+lo[k],1+hi[k]) if k==1 else (-hi[k],-lo[k])
            zmax=max(abs(g*(r+c)) for g in (gl,gh) for r in (r0,r1) for c in (cl,ch))
            v*=aa(gl*(ch-cl))*(-aa(zmax*zmax)/2).exp()/pi2.sqrt()
        v*=aa((r1-r0)*(gh-gl))
        return v
