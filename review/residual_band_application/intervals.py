"""Exact stored-input affine line partition; no floating topology decisions."""
from fractions import Fraction as F

class Ledger:
    def __init__(self):
        self.counts={}; self.max_returned_num_bits=0; self.max_returned_den_bits=0
    def add(self,key,n=1):self.counts[key]=self.counts.get(key,0)+n
    def observe(self,x):
        self.max_returned_num_bits=max(self.max_returned_num_bits,abs(x.numerator).bit_length())
        self.max_returned_den_bits=max(self.max_returned_den_bits,x.denominator.bit_length());return x
    def mul(self,a,b):self.add('rational_multiply');return self.observe(a*b)
    def plus(self,a,b):self.add('rational_add');return self.observe(a+b)
    def div(self,a,b):self.add('rational_divide');return self.observe(a/b)
    def neg(self,a):self.add('rational_negate');return self.observe(-a)
    def affine(self,terms):
        c=s=F(0)
        for w,(a,b) in terms:c=self.plus(c,self.mul(w,a));s=self.plus(s,self.mul(w,b))
        return c,s

def validate(model,y,d,lo,hi):
    wi,wh,wo=model;H=len(wi)
    if not H or len(wh)!=H or any(len(r)!=H for r in wh) or not wo or any(len(r)!=H for r in wo):raise ValueError('model shape')
    if not y or len(d)!=len(y) or len(wo)!=len(y):raise ValueError('input/output shape')
    values=list(wi)+[v for r in wh for v in r]+[v for r in wo for v in r]+list(y)+list(d)+[lo,hi]
    if any(type(x)is not F for x in values):raise ValueError('exact Fraction inputs required')
    if not lo<hi or not lo<=0<=hi:raise ValueError('window must straddle or touch zero')

def partition(model,y,d,lo,hi,ledger,frozen=False,max_pieces=10000):
    validate(model,y,d,lo,hi);wi,wh,wo=model;H=len(wi)
    pieces=[(lo,hi,[(F(0),F(0)) for _ in wi])];layers=[]
    for t in range(len(y)):
        new=[]
        for a,b,h in pieces:
            z=[]
            for i in range(H):
                c,s=ledger.affine(zip(wh[i],h));z.append((ledger.plus(c,ledger.mul(wi[i],y[t])),ledger.plus(s,ledger.mul(wi[i],d[t]))))
            cuts={a,b}
            if not frozen:
                for c,s in z:
                    ledger.add('root_tests')
                    if s:
                        root=ledger.div(ledger.neg(c),s)
                        if a<root<b:cuts.add(root)
            cuts=sorted(cuts);ledger.add('sort_items',len(cuts))
            for left,right in zip(cuts,cuts[1:]):
                mid=F(0) if frozen else ledger.div(ledger.plus(left,right),F(2))
                hh=[];word=[]
                for c,s in z:
                    on=ledger.plus(c,ledger.mul(s,mid))>0;ledger.add('gate_sign_tests');word.append(int(on));hh.append((c,s) if on else (F(0),F(0)))
                new.append((left,right,hh));ledger.add('piece_outputs')
                if len(new)>max_pieces:raise RuntimeError('piece cap')
        pieces=new;layers.append(len(pieces))
    cells=[]
    for a,b,h in pieces:cells.append((a,b,[ledger.affine(zip(row,h)) for row in wo]))
    return cells,layers

def merge(intervals):
    out=[]
    for a,b in sorted(intervals):
        if not a<b:raise ValueError('nonpositive interval')
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
        else:out.append((a,b))
    return out

def errors(cells,truth,lo,hi,ledger):
    out=[]
    for left,right,logits in cells:
        a=max(lo,left);b=min(hi,right)
        if a>=b:continue
        if type(truth)is not int or not 0<=truth<len(logits):raise ValueError('truth')
        ca,cb=a,b;ok=True
        for k,(c,s) in enumerate(logits):
            if k==truth:continue
            c=ledger.plus(logits[truth][0],ledger.neg(c));s=ledger.plus(logits[truth][1],ledger.neg(s));ledger.add('margin_tests')
            if s==0:
                if c<0 or (c==0 and k<truth):ok=False;break
            else:
                root=ledger.div(ledger.neg(c),s)
                if s>0:ca=max(ca,root)
                else:cb=min(cb,root)
        if not ok or ca>=cb:out.append((a,b));continue
        if a<ca:out.append((a,ca))
        if cb<b:out.append((cb,b))
    return merge(out)

def signed_difference(actual,frozen,ledger):
    cuts=sorted({x for pair in actual+frozen for x in pair});out=[]
    for a,b in zip(cuts,cuts[1:]):
        mid=ledger.div(ledger.plus(a,b),F(2));ledger.add('membership_interval_tests',len(actual)+len(frozen))
        sign=int(any(l<mid<r for l,r in actual))-int(any(l<mid<r for l,r in frozen))
        if sign:
            if out and out[-1][1]==a and out[-1][2]==sign:out[-1]=(out[-1][0],b,sign)
            else:out.append((a,b,sign))
    return out
