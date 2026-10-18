from pathlib import Path
from fractions import Fraction as F
import ast,json,hashlib,math
P=Path('/workspace/work/recover-small_h2_34/review/srb_cost_review/composition')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((P/'protocol.json').read_text())
for path,h in p['pins'].items():assert sha(path)==h,path
f=json.loads((P/'fixtures.json').read_text());assert len(f['cases'])==4 and [(c['budget'],c['delayed']) for c in f['cases']]==[(0,False),(1000,False),(1000,True),(17,False)]
labels=[];edges=list(map(F,[0.,.01,.025,.05,.1,.2]))
for raw in f['prototypes']:
 x=sorted(map(F,raw),reverse=True);a,b,c=x[:3];gap=min((a-b)/abs(a),(b-c)/max(abs(b),F(1e-12)))
 label=0 if b<=0 else min(6,sum(gap>=e for e in edges));labels.append(label)
assert labels==list(range(7))
a=json.loads(Path(p['allocation']).read_text())['plain'];cases=[]
for c in f['cases']:
 n=int(c['budget']//(1.6+.1));nh=[max(4,math.floor(v*n)) for v in a];assert max(nh)<50000//7
 M=sum(nh);D=100000 if c['delayed'] else 50000
 cases.append(dict(name=c['name'],n=n,nh=nh,M=M,D=D,raw_cost=math.ceil(M*(1.6+.1)),extra_nominal=str(F(D-M,25))))
tree=ast.parse(Path(p['cv_source']).read_text());selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'pstratum','pc','draw_all','make'}];assert len(selected)==4
# The loader takes the original nodes, only supplying reached globals; no source rewrite.
source=Path(p['q_source']).read_text();assert 'COST_Q = 1.6' in source
h=ast.parse(Path(p['harness_source']).read_text());assert len([n for n in h.body if isinstance(n,ast.FunctionDef) and n.name=='_charged'])==1
print(json.dumps(dict(pins=len(p['pins']),prototype_labels=labels,cases=cases,author_function_calls=0)))
