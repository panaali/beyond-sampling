import ast,json,hashlib
from pathlib import Path
from fractions import Fraction as F
P=Path('/workspace/work/recover-small_h4_4/review/srb_tie_semantics')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((P/'protocol.json').read_text())
for path,h in p['pins'].items():assert sha(path)==h,path
s=(P/'rb9_snapshot.py').read_text();t=ast.parse(s);fs=[n for n in t.body if isinstance(n,ast.FunctionDef)]
assert [n.name for n in fs]==['h9','Q']
a=ast.parse((P/'adapter.py').read_text());constants={n.targets[0].id:ast.literal_eval(n.value) for n in a.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant)}
s='\n\n'.join(ast.get_source_segment(s,n) for n in fs)
assert s.count(constants['OLD'])==1
new=s.replace(constants['OLD'],constants['NEW']);assert ast.dump(ast.parse(new.replace(constants['NEW'],constants['OLD'])))==ast.dump(ast.parse(s))
fixtures=json.loads((P/'fixtures.json').read_text());assert len(fixtures)==6
preds=[]
for f in fixtures:
 assert all(all(v==0 for v in row) for row in f['Wh'])
 assert len(f['Wo'])==10 and len(f['Wh'])==len(f['Wi'])
 assert len(f['x'])==2 and all(len(x)==len(set(x))==9 for x in f['x'])
 # With Wh=0, last logits on either open sign halfline are these scalar slopes times |y|.
 winners=[]
 for sign in [-1,1]:
  coeff=[sum(F(v)*max(F(0),sign*F(w)) for v,w in zip(row,f['Wi'],strict=True)) for row in f['Wo']]
  winners.append(max(range(10),key=lambda j:(coeff[j],-j)))
 kind=f['prediction'];expected=[int(kind[-1])]*2 if kind.startswith('constant') else [1,0] if kind=='negative1_positive0' else [0,5]
 assert winners==expected
 preds.append({'name':f['name'],'negative_positive_predictions':winners})
print(json.dumps({'pins':len(p['pins']),'fixture_predictions':preds,'restored_AST':True,'calls':p['Q_calls']}))
