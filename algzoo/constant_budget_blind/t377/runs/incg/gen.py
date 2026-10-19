import json
sp={int(k):(v[1],[round(c,4) for c in v[2]]) for k,v in json.load(open("final_spec.json")).items()}
sp[25]=(["S0","x9"],[-4.076,-0.54])
sp[16]=(["isMax7","S0"],[10.442,-1.209])
L={
0:("B","Mostly fires when x0 is among the two largest inputs (scaled by x0's lead over the third largest of the rest); also carries a baseline that rises with the overall top values."),
1:("A","Rank detector for x2: big when x2 is the maximum (grows with its lead over the runner-up), moderate when x2 is second (lead over third), plus a term in max(x0,x2)."),
2:("A","Rank detector for x4: big when x4 is the maximum (grows with its lead over the runner-up), moderate when x4 is second (lead over third), plus a term in max(x0,x4)."),
3:("A","How far x9 falls below the largest of x0..x8, reduced (by the max's lead over the runner-up) when that max sits at x0,x2,x3,x5,x6 or x8, and increased when it is x1."),
4:("B","Lead of the largest of x2..x8 over the second-largest of x0..x8 (max-position code favouring middle positions), with a bonus when x5 is in the top two and a penalty when x2 is."),
5:("D","Weak, small signal: roughly the max of x1..x6 minus a bit of x0, plus extra when x2 is the maximum; switched off whenever x7 exceeds all earlier inputs (and when x0 is the maximum)."),
6:("A","Max-position code: positive when the overall maximum lies in x4..x8 (and beats x9), proportional to its lead over earlier maxima and the runner-up, stronger for later positions."),
7:("A","How far x9 is below the largest of x0..x7: about 5 times ReLU(max(x0..x7) - x9). Zero when x9 exceeds all of x0..x7."),
8:("A","x9 is the new overall maximum: about 6.8*x9 - 6.5*max(x0..x8), passed through ReLU; zero unless x9 is (nearly) the largest input."),
9:("A","How far x9 falls below the largest of x0..x8 (about 4.3 times the gap), minus the max's lead over the runner-up when the max is at x2 or x3."),
10:("D","Rare, small: fires mostly when x8 is the overall maximum and the runner-up is early (x0..x5), larger when the runner-up is close to x8; formula only partly captures it."),
11:("A","Rank detector for x3: big when x3 is the maximum (grows with its lead over the runner-up), moderate when x3 is second (lead over third), plus a term in max(x0,x3)."),
12:("B","Max-position code: lead of the largest of x2,x3,x4,x6,x7 over second-ranked values; zero when x5 is the maximum, and partly tracks how far x5 sits below the maximum."),
13:("A","x9 is among the two largest inputs: about 5.9 times x9's lead over the second-largest of x0..x8, reduced when x0 or x1 is also in the top two."),
14:("A","How far x8 falls below the largest other input (about 5.4 times the gap), reduced when x9 is the maximum or when the max lies in x4..x9."),
15:("A","Max-position code favouring early maxima: large when the overall maximum is early (especially x0, or within x0..x5) and well above later values; small when the maximum comes late."),
16:("D","Essentially silent on Gaussian inputs: fires only when every input is negative and x7 is the maximum, growing with x7's lead and with how negative the maximum is."),
17:("A","How far x9 falls below the largest of x0..x8, reduced by the max's lead over the runner-up when that max lies in x4..x8 (mildly for x0,x1,x3), increased when the max is x2."),
18:("A","How far x9 falls below the largest of x0..x8 (about 4.5 times the gap), shifted up when the max sits at x0,x5,x7 or x8 and down when it is at x1..x4 or x6."),
19:("A","Overall level of the top inputs: about 10*max(x0..x8) + 5.3*second-largest input, plus extra for the maximum's lead when the maximum is x0 or x1."),
20:("A","How far x9 falls below the largest of x0..x8, minus the max's lead over the runner-up when the max sits at x1,x2,x5,x6,x7 or x8 (so mostly kept for maxima at x0,x3,x4)."),
21:("A","Rank detector for x1: very big when x1 is the maximum (grows with its lead over the runner-up), moderate when x1 is second (lead over third), plus a term in max(x0,x1)."),
22:("A","How far x9 falls below the largest of x0..x8 (about 4.2 times the gap), minus the max's lead over the runner-up when the max is at x0,x2,x4,x5 (half that at x1); kept for x3,x6..x8."),
23:("A","Rank detector for x6: grows with x6 itself and with x6's lead over the third-largest of the other inputs (x6 among the top two), minus the second-largest of x0..x5."),
24:("B","How far x7 falls below the largest other input, plus 2.8*max(x0..x6) - 1.8*max(x2..x9), with bonuses when x8 is among the top two or x0 is the maximum."),
25:("D","Essentially silent on Gaussian inputs: fires only when every input is negative, proportional to how far the maximum (and x9) lies below zero."),
26:("B","Two signals: x7 is large relative to the second-largest of x0..x6, and x0 is among the two largest inputs (lead over third); reduced when x7 is a new running maximum."),
27:("A","x8 is a new running maximum: about 8.4*x8 - 7.9*max(x0..x7), passed through ReLU; zero unless x8 beats all earlier inputs."),
28:("A","How far x9 falls below the largest of x0..x8, minus the max's lead over the runner-up when the max is at x2,x3 or x7, plus it when the max is x1."),
29:("B","x0/x1 signal: rises with max(x0,x1), peaks when x0 is the second-largest input (in top two but not the maximum), and is reduced when x0 or x1 is the clear maximum."),
30:("A","How far x9 falls below the largest of x0..x8, with the largest of x0,x2,x4,x6,x7,x8 weighted much more than that of x1,x3,x5 (max at odd early positions counts less)."),
31:("B","Second-largest value among x0..x7 (minus small corrections), switched off when x8 is among the two largest inputs."),
}
ST={i:"" for i in range(32)}
for i,(nm,c) in sp.items():
    ST[i]="features chosen by greedy forward selection / partition search over a library: "+", ".join(nm)
ST[16]="gate: output 0 unless max(x)<0; features isMax7,S0 chosen by hand"
ST[25]="features S0,x9 chosen from a small candidate set"
src='''import numpy as np
# Feature glossary (x is (N,10)):
#  xj: input j; Sr: r-th largest input (S0=max); pmT: max(x0..xT); smT: max(xT..x9); p2ndT: 2nd largest of x0..xT
#  mexK: max of inputs other than xK; s2exK: 2nd largest of others; isMaxK=ReLU(xK-mexK); inTop2_K=ReLU(xK-s2exK)
#  belowK=ReLU(mexK-xK); recK=ReLU(xK-max(x0..xK-1)); gmaxABC: max of x_A,x_B,x_C
# Each neuron: f = ReLU(sum_k c_k * feature_k)  (no intercept; the model is positively homogeneous)
def _F(x,n):
    if n.startswith("inTop2_"):
        k=int(n[7:]); o=np.sort(np.delete(x,k,1),1); return np.maximum(x[:,k]-o[:,-2],0)
    if n.startswith("isMax"):
        k=int(n[5:]); return np.maximum(x[:,k]-np.delete(x,k,1).max(1),0)
    if n.startswith("below"):
        k=int(n[5:]); return np.maximum(np.delete(x,k,1).max(1)-x[:,k],0)
    if n.startswith("rec"):
        k=int(n[3:]); return np.maximum(x[:,k]-x[:,:k].max(1),0)
    if n.startswith("mex"):
        k=int(n[3:]); return np.delete(x,k,1).max(1)
    if n.startswith("s2ex"):
        k=int(n[4:]); return np.sort(np.delete(x,k,1),1)[:,-2]
    if n.startswith("gmax"): return x[:,[int(c) for c in n[4:]]].max(1)
    if n.startswith("p2nd"):
        t=int(n[4:]); return np.sort(x[:,:t+1],1)[:,-2]
    if n.startswith("pm"):
        t=int(n[2:]); return x[:,:t+1].max(1)
    if n.startswith("sm"):
        t=int(n[2:]); return x[:,t:].max(1)
    if n.startswith("S"):
        r=int(n[1:]); return -np.sort(-x,1)[:,r]
    if n.startswith("x"): return x[:,int(n[1:])]
    raise KeyError(n)
'''
src+="SPEC = "+repr({i:sp[i] for i in range(32)})+"\n"
src+="LABELS = "+repr(L)+"\n"
src+="CONSTANTS = "+repr({i:len(sp[i][1]) for i in range(32)})+"\n"
src+="STRUCT = "+repr(ST)+"\n"
src+='''
def f(i, x):
    x=np.asarray(x,dtype=float)
    names,coefs=SPEC[i]
    z=np.zeros(len(x))
    for n,c in zip(names,coefs): z=z+c*_F(x,n)
    out=np.maximum(z,0)
    if i==16: out=np.where(x.max(1)<0,out,0.0)
    return out
'''
open("labels.py","w").write(src)
for i in range(32): assert len(L[i][1].split())<=40,(i,len(L[i][1].split()))
print("ok")
