"""Independent packed-byte T25 parser and stdlib NPY reader. No author imports."""
import ast, hashlib, math, struct, zipfile
TYPES=('none','bool','int','str','float','list','ast','tuple')
NAMES=sorted(n for n in dir(ast) if isinstance(getattr(ast,n),type) and issubclass(getattr(ast,n),ast.AST))
SCHEMA={n:list(getattr(ast,n)._fields) for n in NAMES}

def same(a,b,json_containers=False):
    if isinstance(a,ast.AST):
        assert type(a) is type(b)
        for f in a._fields:same(getattr(a,f,None),getattr(b,f,None),json_containers)
    elif type(a) in (list,tuple):
        assert type(b) in ((list,tuple) if json_containers else (type(a),)) and len(a)==len(b)
        for x,y in zip(a,b,strict=True):same(x,y,json_containers)
    else:
        assert type(a) is type(b),(type(a),type(b))
        if type(a) is float:assert math.isfinite(a) and struct.pack('>d',a)==struct.pack('>d',b)
        else:assert a==b

class Reader:
    def __init__(self,data,bits):
        assert type(bits)is int and 0<bits<=80_000_000 and len(data)==(bits+7)//8
        self.data=data;self.end=len(data)*8;self.pos=self.end-bits
        assert self.pos<8 and data[0]>>(8-self.pos)==0
        self.counts={k:0 for k in TYPES}
    def take(self,n):
        assert n>=0 and self.pos+n<=self.end
        lo=self.pos;self.pos+=n
        if not n:return 0
        stop=(self.pos+7)//8
        return (int.from_bytes(self.data[lo//8:stop],'big')>>(8*stop-self.pos))&((1<<n)-1)
    def natural(self):
        zeros=0
        while not self.take(1):
            zeros+=1;assert zeros<63
        return (1<<zeros)+self.take(zeros)
    def value(self,depth=0):
        assert depth<1000
        tag=self.take(3);self.counts[TYPES[tag]]+=1
        if tag==0:return None
        if tag==1:return bool(self.take(1))
        if tag==2:
            z=self.natural()-1;return -(z//2)-1 if z&1 else z//2
        if tag==3:
            n=self.natural()-1;assert n*8<=self.end-self.pos
            return bytes(self.take(8) for _ in range(n)).decode('utf8')
        if tag==4:
            v=struct.unpack('>d',self.take(64).to_bytes(8,'big'))[0];assert math.isfinite(v);return v
        if tag in (5,7):
            n=self.natural()-1;assert n<=(self.end-self.pos)//3
            a=[self.value(depth+1) for _ in range(n)];return a if tag==5 else tuple(a)
        code=self.take(8);assert code<len(NAMES);cls=getattr(ast,NAMES[code])
        return cls(**{f:self.value(depth+1) for f in cls._fields})
    def piece(self):
        start=self.pos;before=self.counts.copy();v=self.value()
        return v,{'bits':self.pos-start,'counts':{k:self.counts[k]-before[k] for k in TYPES}}
    def finish(self):assert self.pos==self.end

def from_bittext(bits):
    assert bits and set(bits)<={'0','1'}
    r=Reader(int(bits,2).to_bytes((len(bits)+7)//8,'big'),len(bits));v,m=r.piece();r.finish();return v,m

def program(root,kind):
    # Rebuild only syntax; never execute author wrappers or Normalize.
    def parse(name):return ast.parse((root/name).read_text())
    if kind=='tables':
        t=parse('frozen/features.py')
        t.body=[n for n in t.body if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,ast.FunctionDef) and n.name in {'_array','features'}]
        t.body.extend(parse('inference.py').body)
    elif kind=='network':t=parse('network.py')
    else:
        assert kind=='HA1';t=parse('frozen/labels.py')
        t.body=[n for n in t.body if not(isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='LABELS' for v in n.targets))]
        t.body.extend(parse('ha_wrapper.py').body)
    def clean(v):
        if isinstance(v,list):return [clean(x) for x in v]
        if not isinstance(v,ast.AST):return v
        for f in v._fields:
            if hasattr(v,f):setattr(v,f,clean(getattr(v,f)))
        if isinstance(v,(ast.Module,ast.FunctionDef)):
            v.body=[n for n in v.body if not(isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and type(n.value.value)is str)]
        if isinstance(v,ast.UnaryOp) and isinstance(v.op,ast.USub) and isinstance(v.operand,ast.Constant) and type(v.operand.value)is float:
            return ast.Constant(value=-v.operand.value)
        return v
    return clean(t)

def array_from_npz(path,key,record):
    with zipfile.ZipFile(path) as z:
        assert set(z.namelist())=={k+'.npy' for k in record['arrays']}
        raw=z.read(key+'.npy')
    assert raw[:6]==b'\x93NUMPY';version=raw[6:8]
    assert version in (b'\x01\x00',b'\x02\x00')
    nbytes=2 if version[0]==1 else 4;h=8+nbytes
    size=int.from_bytes(raw[8:h],'little');meta=ast.literal_eval(raw[h:h+size].decode('latin1'));body=raw[h+size:]
    assert set(meta)=={'descr','fortran_order','shape'} and meta['fortran_order'] is False
    desc=meta['descr'];assert desc in ('<f8','<i8')
    shape=meta['shape'];assert type(shape)is tuple and all(type(n)is int and n>=0 for n in shape)
    expected=record['arrays'][key]
    assert list(shape)==expected['shape'] and expected['dtype']==('float64' if desc=='<f8' else 'int64')
    assert len(body)==8*math.prod(shape) and hashlib.sha256(body).hexdigest()==expected['sha256']
    vals=[x[0] for x in struct.iter_unpack('<d' if desc=='<f8' else '<q',body)]
    if desc=='<f8':assert all(math.isfinite(x) for x in vals)
    def nest(offset,dims):
        if len(dims)==1:return vals[offset:offset+dims[0]]
        step=math.prod(dims[1:]);return [nest(offset+i*step,dims[1:]) for i in range(dims[0])]
    assert shape
    return nest(0,shape),shape,desc
