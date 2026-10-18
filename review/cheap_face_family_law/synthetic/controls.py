"""One gated fixed synthetic control run. No source face tables or network calls."""
import hashlib,json,os,re,time,traceback
from fractions import Fraction as F
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'output'
START=time.process_time()

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,obj):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');q.replace(p)
def pins(mapping):
    assert mapping
    for path,h in mapping.items():assert sha(path)==h,path

def main():
    assert not OUT.exists(),'No overwrite or silent retry'
    gate=read(HERE/'gate.json');protocol=read(HERE/'protocol.json')
    assert gate['status']=='ready' and re.fullmatch('[0-9a-f]{8,40}',gate['review_publication'])
    rp=Path(gate['source_review']);rv=read(rp)
    assert rv['status']=='passed' and rv['execution_authorized'] is True
    need={str(HERE/n) for n in ['primitive.py','controls.py','fixtures.json','protocol.json']}
    assert need <= set(rv['pins']) and need|{str(rp)} <= set(gate['pins'])
    pins(rv['pins']);pins(gate['pins']);pins(protocol['pins'])
    f=read(HERE/'fixtures.json')
    assert len(f['grids'])==8 and len(f['guards'])==16
    assert len({x['name'] for x in f['grids']+f['guards']})==24
    OUT.mkdir()
    result={'status':'running','pins':gate['pins'],'rows':[],'guard_rows':[],'conditional_calls':0,'model_grid_face_rng_calls':0}
    def save():result['cpu']=time.process_time()-START;put(OUT/'results.json',result)
    def event(kind,**kw):
        with (OUT/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps({'event':kind,'cpu':time.process_time()-START,**kw})+'\n');stream.flush();os.fsync(stream.fileno())
    try:
        from primitive import Grid,CUTS
        event('start');save()
        for spec in f['grids']:
            assert time.process_time()-START<16
            row={'name':spec['name'],'status':'started','fixture':spec,'cdf':[],'inverse':[],'conditional':[]}
            result['rows'].append(row);save();event('grid_start',name=spec['name'])
            x=tuple(map(F,spec['x']));c=tuple(map(F,spec['c']));g=Grid(x,c)
            b=g.bins();row['bins']=list(map(str,b));save()
            assert b==tuple(map(F,spec['bins'])) and sum(b)==1 and min(b)>=0
            # Full fixed knot coverage plus all three bin boundaries.
            for u in (*x,*CUTS):
                value=g.cdf(u);row['cdf'].append({'u':str(u),'value':str(value)});save()
                if u in x:assert value==c[x.index(u)]
            for qs,expected in zip(f['q'],spec['inverse'],strict=True):
                q=F(qs);u=g.inverse(q);row['inverse'].append({'q':str(q),'u':str(u)});save()
                assert u==F(expected)
            for j,p in enumerate(b):
                if p==0:continue
                for vs in f['conditional_v']:
                    v=F(vs);got=g.conditional(j,v);result['conditional_calls']+=1
                    row['conditional'].append({'bin':j,'v':str(v),**{key:str(val) for key,val in got.items()}});save()
                    u=got['u'];label=0 if u<=0 else 1 if u<=F(1,4) else 2 if u<=F(1,2) else 3
                    assert got['probability']==p and label==j
                    left=sum(b[:j]);assert got['q']==left+p*v
            row['status']='passed';save();event('grid_complete',name=spec['name'])
        for spec in f['guards']:
            assert time.process_time()-START<16
            row={'name':spec['name'],'fixture':spec,'status':'started'};result['guard_rows'].append(row);save();event('guard_start',name=spec['name'])
            caught=None
            try:
                op=spec['op']
                if op=='construct':Grid(tuple(map(F,spec['x'])),tuple(map(F,spec['c'])))
                elif op=='float_construct':Grid((0.0,F(1)),(F(0),F(1)))
                else:
                    g=Grid((F(1),F(2)) if op=='zero_stratum' else (F(-1),F(1)),(F(0),F(1)))
                    if op=='inverse':g.inverse(F(spec['q']))
                    elif op=='float_cdf':g.cdf(0.0)
                    elif op=='conditional':g.conditional(spec['bin'],F(spec['v']))
                    elif op=='zero_stratum':g.conditional(0,F(1,2))
                    else:raise AssertionError('unknown frozen guard')
            except (ValueError,TypeError) as exc:caught=type(exc).__name__;row['exception']=repr(exc)
            row['caught']=caught;save();assert caught==spec['error']
            row['status']='passed';save();event('guard_complete',name=spec['name'])
        assert result['conditional_calls']==78
        result['status']='complete';save();event('complete')
    except BaseException as exc:
        result['status']='failed';result['error']=repr(exc);result['traceback']=traceback.format_exc();save();put(OUT/'failure.json',result);event('failed',error=repr(exc));raise

if __name__=='__main__':main()
