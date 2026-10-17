"""Exact projective gate cells for a fixed word, including recession rays."""
from fractions import Fraction as F
import time

ZERO=F(0);ONE=F(1)

def rational(v):return v if isinstance(v,F) else F(v)

class Geometry:
    def __init__(self,wi,wh,seconds=100):
        self.wi=tuple(map(rational,wi));self.wh=tuple(tuple(map(rational,row)) for row in wh)
        if len(self.wi)!=2 or any(len(row)!=2 for row in self.wh):raise ValueError('H2 only')
        self.stats={'rational_mul':0,'rational_add_sub':0,'rational_div':0,'dot_calls':0,'clips':0,'created_vertices':0,'direct_steps':0,'max_fraction_bits':0}
        self.start=time.process_time();self.seconds=seconds

    def dot(self,a,b):
        self.stats['dot_calls']+=1;self.stats['rational_mul']+=len(a);self.stats['rational_add_sub']+=len(a)-1
        return sum((x*y for x,y in zip(a,b)),ZERO)

    def area2(self,p):
        self.stats['rational_mul']+=2*len(p);self.stats['rational_add_sub']+=2*len(p)
        return sum((u[1]*v[2]-u[2]*v[1] for u,v in zip(p,p[1:]+p[:1])),ZERO)

    def clip(self,p,values,sign):
        self.stats['clips']+=1;out=[]
        for i,u in enumerate(p):
            j=(i+1)%len(p);v=p[j];a=sign*values[i];b=sign*values[j]
            if a>=0:out.append(u)
            if (a<0<b) or (b<0<a):
                frac=a/(a-b);self.stats['rational_div']+=1;self.stats['rational_add_sub']+=7;self.stats['rational_mul']+=3
                out.append(tuple(u[k]+frac*(v[k]-u[k]) for k in range(3)))
                self.stats['created_vertices']+=1
        clean=[]
        for v in out:
            if not clean or v!=clean[-1]:clean.append(v)
        if len(clean)>1 and clean[0]==clean[-1]:clean.pop()
        if len(clean)<3:return []
        area=self.area2(clean)
        if area<0:raise AssertionError('orientation reversed')
        return clean if area else []

    def direct(self,initial,word):
        h=tuple(initial)
        for x in word:
            h=tuple(max(self.dot(row,h)+w*x,ZERO) for row,w in zip(self.wh,self.wi))
            self.stats['rational_mul']+=2;self.stats['rational_add_sub']+=2;self.stats['direct_steps']+=1
        return h

    def propagate(self,word):
        word=tuple(map(rational,word));p=[(ONE,ZERO,ZERO),(ZERO,ONE,ZERO),(ZERO,ZERO,ONE)]
        cells=[(p,((ZERO,ONE,ZERO),(ZERO,ZERO,ONE)),None)]
        trace=[]
        for step,x in enumerate(word):
            nxt=[]
            for p,B,_ in cells:
                if time.process_time()-self.start>self.seconds:raise RuntimeError('CPU guard, no result certified')
                Z=tuple(tuple(self.dot(self.wh[i],(B[0][j],B[1][j]))+(self.wi[i]*x if j==0 else ZERO) for j in range(3)) for i in range(2))
                self.stats['rational_mul']+=2;self.stats['rational_add_sub']+=6
                parts=[(p,((ZERO,ZERO,ZERO),(ZERO,ZERO,ZERO)))]
                for i in range(2):
                    split=[]
                    for q,C in parts:
                        vals=[self.dot(Z[i],v) for v in q]
                        if max(vals)<=0:split.append((q,C))
                        elif min(vals)>=0:
                            D=list(C);D[i]=Z[i];split.append((q,tuple(D)))
                        else:
                            for sign in (-1,1):
                                r=self.clip(q,vals,sign)
                                if r:
                                    D=list(C)
                                    if sign==1:D[i]=Z[i]
                                    split.append((r,tuple(D)))
                    parts=split
                nxt.extend((q,C,Z) for q,C in parts)
            cells=nxt
            total=sum((self.area2(q) for q,_,_ in cells),ZERO)
            if total!=1:raise AssertionError(('partition area',step,total))
            trace.append({'step':step+1,'cells':len(cells),'vertices':sum(len(q) for q,_,_ in cells),'area2':str(total)})
        return cells,trace,word

    def check(self,word,validate=True):
        cells,trace,word=self.propagate(word);reset=True;witness=None;eta=[None,None];unbounded=[False,False];recession=0;points=0
        for p,B,Z in cells:
            coefficients=[x for row in B for x in row]+([x for row in Z for x in row] if Z else [])
            self.stats['max_fraction_bits']=max(self.stats['max_fraction_bits'],*(max(x.numerator.bit_length(),x.denominator.bit_length()) for x in coefficients))
            finite=next(v for v in p if v[0]>0)
            for v in p:
                self.stats['max_fraction_bits']=max(self.stats['max_fraction_bits'],*(max(x.numerator.bit_length(),x.denominator.bit_length()) for x in v))
                image=tuple(self.dot(row,v) for row in B)
                if min(image)<0:raise AssertionError('negative hidden image')
                recession+=v[0]==0;points+=1
                if any(image):
                    reset=False
                    if witness is None:
                        u=v if v[0]>0 else tuple(v[i]+finite[i] for i in range(3))
                        h=(u[1]/u[0],u[2]/u[0]);actual=self.direct(h,word)
                        if not any(actual):raise AssertionError('counterexample failed')
                        witness={'initial':[str(x) for x in h],'final':[str(x) for x in actual],'from_recession_vertex':v[0]==0}
                if Z is not None:
                    for i,row in enumerate(Z):
                        value=self.dot(row,v)
                        if v[0]:
                            candidate=-value/v[0];eta[i]=candidate if eta[i] is None else min(eta[i],candidate)
                            self.stats['rational_div']+=1
                        elif value>0:unbounded[i]=True
                if validate and v[0]:
                    actual=self.direct((v[1]/v[0],v[2]/v[0]),word)
                    if actual!=tuple(z/v[0] for z in image):raise AssertionError('vertex map mismatch')
            if validate:
                center=tuple(sum((v[i] for v in p),ZERO) for i in range(3))
                actual=self.direct((center[1]/center[0],center[2]/center[0]),word)
                image=tuple(self.dot(row,center)/center[0] for row in B)
                if actual!=image:raise AssertionError('centroid map mismatch')
        C=[ZERO,ZERO];v=list(map(abs,self.wi));A=[list(map(abs,row)) for row in self.wh]
        for _ in word:
            C=[a+b for a,b in zip(C,v)];v=[self.dot(row,v) for row in A]
        eta=[None if flag else e for flag,e in zip(unbounded,eta)]
        radii=[e/c for e,c in zip(eta,C) if c and e is not None]
        radius=min(radii) if reset and radii and all(e is not None and e>0 for e in eta) else ZERO
        return {'word':[str(x) for x in word],'global_reset':reset,'robust_radius':str(radius),'eta':[str(e) for e in eta],'unbounded_preactivation':unbounded,'lipschitz':[str(c) for c in C],'counterexample':witness,'trace':trace,'final_cells':len(cells),'checked_vertices':points,'recession_vertices':recession,'stats':dict(self.stats),'cpu_seconds':time.process_time()-self.start}
