p='labels.py'; s=open(p).read()
cut=s.index('LABELS = ')
s=s[:cut]
def x9neg(P,gate):
    g=" and the maximum is not x8" if gate else ""
    return ("Active when x9 is below the second-largest of x0..x8%s: a mix of the top two of x0..x8 minus 1.5-1.9 times x9, reweighted when the maximum is at %s." % (g, ", ".join("x%d"%q for q in P)))
def rk(p,extra=""):
    return "Fires when x%d is among the three largest inputs: a rank-dependent multiple of x%d (largest > second > third), minus about 0.15-0.25 times the sum of the other nine inputs.%s" % (p,p,extra)
def notp(p,k,extra):
    w="one of the two largest" if k==2 else "the largest"
    return "Silent when x%d is %s; otherwise roughly 0.5 x max + 0.8-1.2 x second-largest - 0.25-0.5 x x%d%s." % (p,w,p,extra)
L={
0:("A",x9neg([0,3,4,7],True)),
1:("A","Gap between the largest and second-largest of x0..x8, zero when x8 itself is the largest of x0..x8: 4.8 x ReLU(max(x0..x7) - second-largest of x0..x8)."),
2:("A",rk(3)),
3:("D","Unclear. Loosely follows max(x0..x4) minus the second-largest of x0..x4, larger when the second-largest overall sits at x5-x7 and the max earlier; silent when x8 is in the top two."),
4:("D","Weak parity signal: larger when the second-largest input is at an even position (x0,x2,x4,x6,x8) and x9 is not the maximum; scales with the second-largest value. Poorly fit."),
5:("A","Always on: gap between largest and second-largest of x0..x8, plus extra when x8 sets a new running maximum (x8 minus max of x0..x7), plus a max(x0..x4) term."),
6:("A",x9neg([0,2,5,7],False)),
7:("A","How much x9 exceeds every earlier input: 8.8 x ReLU(x9 - max(x0..x8)); zero unless x9 is the largest."),
8:("C","Always on: generic magnitude (sum of |x|, minus sum of x, running maxima of x0..x2 and x0..x5) plus a strong boost when x5 is one of the two largest inputs."),
9:("A","Fires only when inputs are mostly negative: ReLU of -(2.5 x second-largest of x0..x8 + 0.8 x third-largest of x0..x7 + 0.2 x x9)."),
10:("A","Small signal that the maximum is at x4 or x5 (about 0.5 x the max), suppressed when x7 or x8 is among the two largest."),
11:("A","Codes the maximum's position: about 1.3 x max if the max is x4, 0.7 x max if it is x0 or x6, 0.2 x max if it is x9, otherwise zero."),
12:("A","How far x9 falls below the second-largest of x0..x8: 3.1 x ReLU(second-largest(x0..x8) - x9); zero when x9 is one of the two largest."),
13:("A",x9neg([0,3,5,6],True)),
14:("A","Tracks x8 being one of the two largest: then about 1.2 x the other top-two value + 0.3 x third-largest - 0.28 x sum(x1..x7); otherwise weak ReLU(1.4 x8 - 0.28 sum(x1..x7))."),
15:("A",rk(1)),
16:("A","5.7 x ReLU(x9 - third-largest input): positive exactly when x9 is one of the two largest, measuring its lead over the third."),
17:("A","Magnitude of the top of x0..x8: about 8.1 x max(x0..x8) + 2.0 x second-largest(x0..x8); x9 is ignored."),
18:("A",x9neg([1],False)),
19:("A",rk(6)),
20:("A",notp(0,2,", reweighted when the max is at x1 or x8")),
21:("B","Active when x9 is one of the two largest: about 3.7 x x9 - 2.1 x (other top-two value) - 1.9 x third-largest, boosted when the other top-two input sits at an even position."),
22:("A",rk(7," Strongest when x7 is the maximum.")),
23:("A","4.25 x ReLU(x9 - third-largest input), i.e. x9's lead when it is one of the two largest, but zero when x8 exceeds all of x0..x7."),
24:("A",notp(5,2,", reweighted when the max is at x8 or x9")),
25:("A",notp(8,2,", shrinking the later the max and second-largest occur")),
26:("B","Mainly signals x2 being among the three largest (about 2 x x2, more if x2 is the max); also responds to x0 in the top two and the max at x4, plus a max(x0..x4) term."),
27:("B","Signals the maximum being at x2 or x6 (about 1.17 x max), with small extras when the max is x9 or x2 is in the top three."),
28:("A","Silent when x8 is the maximum; otherwise about max x (1.46 - 0.17 x position of max) + 0.13 x position x second-largest, i.e. larger when the max came early."),
29:("A",x9neg([0,3,6,7],True)),
30:("A",notp(4,2,", reweighted when the max is at x0 or x3")),
31:("A","Second-largest of x0..x8 (about 5x), plus smaller max(x0..x7) and max(x0..x4) terms, reduced when x2 is among the three largest."),
}
for i,(c,t) in L.items():
    assert len(t.split())<=40,(i,len(t.split()),t)
ST={0:"argmax position set {0,3,4,7} found by greedy search; gate argmax!=8",13:"argmax set {0,3,5,6} (greedy); gate argmax!=8",29:"argmax set {0,3,6,7} (greedy); gate argmax!=8",
6:"argmax set {0,2,5,7} (greedy search)",18:"argmax set {1} (greedy search)",
2:"position 3; rank<=2 gating",15:"position 1; rank<=2 gating",19:"position 6; rank<=2 gating",22:"position 7; rank<=2 gating",
20:"p=0, rank>=2 gate, argmax set {1,8} (greedy)",24:"p=5, rank>=2 gate, argmax set {8,9} (greedy)",30:"p=4, rank>=2 gate, argmax set {0,3} (greedy)",
25:"p=8, rank>=2 gate; features chosen by greedy search",28:"p=8, not-max gate; features chosen by greedy search",
3:"5 features chosen by greedy search",8:"5 features chosen by greedy search",9:"4 features chosen by greedy search",10:"positions {4},{5},{7},{8} chosen by greedy",
11:"positions {4},{0,6},{9} chosen by greedy",26:"features chosen by greedy search",27:"positions {2,6},{9},{2},{7} chosen by greedy",31:"features chosen by greedy search",5:"features chosen by greedy search",
14:"feature set chosen by inspection + search",21:"parity of the other top-two position (inspection)",23:"gate: x8 not a new running max",4:"parity of second-largest position; gate argmax!=9",
1:"",7:"",12:"",16:"",17:""}
s+="LABELS = %r\n\nSTRUCT = %r\n\n_x0 = np.random.default_rng(0).standard_normal((20, 10))\nCONSTANTS = {i: len(SPEC[i][1](_base(_x0))) for i in range(32)}\n" % (L,ST)
open(p,'w').write(s)
