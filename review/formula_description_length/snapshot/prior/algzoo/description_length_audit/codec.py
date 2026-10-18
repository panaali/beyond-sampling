"""Frozen T25 decodable syntax code, relative to recorded Python AST schema."""
import ast, math, struct
NAMES=sorted(n for n in dir(ast) if isinstance(getattr(ast,n),type) and issubclass(getattr(ast,n),ast.AST))
assert len(NAMES)<=256
CODES={n:i for i,n in enumerate(NAMES)}
SCHEMA={n:list(getattr(ast,n)._fields) for n in NAMES}
def gamma(n):
 assert n>0
 b=bin(n)[2:];return '0'*(len(b)-1)+b

def encode(x):
 if x is None:return '000'
 if isinstance(x,bool):return '001'+str(int(x))
 if isinstance(x,int):return '010'+gamma((2*x if x>=0 else -2*x-1)+1)
 if isinstance(x,str):
  raw=x.encode();return '011'+gamma(len(raw)+1)+''.join(f'{b:08b}' for b in raw)
 if isinstance(x,float):return '100'+''.join(f'{b:08b}' for b in struct.pack('>d',x))
 if isinstance(x,list):return '101'+gamma(len(x)+1)+''.join(map(encode,x))
 if isinstance(x,ast.AST):return '110'+f'{CODES[type(x).__name__]:08b}'+''.join(encode(getattr(x,f,None)) for f in x._fields)
 if isinstance(x,tuple):return '111'+gamma(len(x)+1)+''.join(map(encode,x))
 raise TypeError(type(x))

class Reader:
 def __init__(self,bits):self.bits=bits;self.i=0
 def take(self,n):
  result=self.bits[self.i:self.i+n];assert len(result)==n;self.i+=n;return result
 def gamma(self):
  z=0
  while self.take(1)=='0':z+=1
  return int('1'+self.take(z),2)
 def read(self):
  tag=self.take(3)
  if tag=='000':return None
  if tag=='001':return bool(int(self.take(1)))
  if tag=='010':
   n=self.gamma()-1;return n//2 if n%2==0 else -(n+1)//2
  if tag=='011':return bytes(int(self.take(8),2) for _ in range(self.gamma()-1)).decode()
  if tag=='100':return struct.unpack('>d',bytes(int(self.take(8),2) for _ in range(8)))[0]
  if tag in ('101','111'):
   out=[self.read() for _ in range(self.gamma()-1)];return out if tag=='101' else tuple(out)
  cls=getattr(ast,NAMES[int(self.take(8),2)]);return cls(**{f:self.read() for f in cls._fields})
def decode(bits):
 r=Reader(bits);out=r.read();assert r.i==len(bits);return out
class Normalize(ast.NodeTransformer):
 def visit_UnaryOp(self,node):
  node=self.generic_visit(node)
  if isinstance(node.op,ast.USub) and isinstance(node.operand,ast.Constant) and isinstance(node.operand.value,float):return ast.copy_location(ast.Constant(value=-node.operand.value),node)
  return node
 def visit_Module(self,node):
  node=self.generic_visit(node);node.body=[n for n in node.body if not(isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))];return node
 def visit_FunctionDef(self,node):
  node=self.generic_visit(node);node.body=[n for n in node.body if not(isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))];return node

def controls():
 fixtures=[None,False,True,-100000,0,100000,'αβ',[],[1,2],(1,'x'),0.0,-0.0,1.25,-.004,ast.parse('x = a + 2\ny = {0, 2, 7}')]
 for x in fixtures:
  y=decode(encode(x))
  if isinstance(x,ast.AST):assert ast.dump(x)==ast.dump(y)
  elif isinstance(x,float):assert struct.pack('>d',x)==struct.pack('>d',y)
  else:assert x==y
 assert encode(ast.parse('x={0,2}'))!=encode(ast.parse('x={0,3}'))
 assert encode(ast.parse('x=a+b'))!=encode(ast.parse('x=a*b'))
 assert len(encode([1,2]))>len(encode([1]))
 return dict(roundtrips=len(fixtures),mutation_controls=3)
