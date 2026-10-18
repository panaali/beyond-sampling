"""Frozen scalar/facet controls and independent positive/radial references."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time,resource
import mpmath as mp
from primitive import prepare,evaluate,radial,mpq,pack
HERE=Path(__file__).resolve().parent

def fhex(x):return F(float.fromhex(x))

def positive_scalar_reference(a,b):
    with mp.workdps(120):
        a=mpq(a);b=abs(mpq(b));calls=0
        if a==0 and b==0:return {'value':'0.0','evaluations':0}
        angle=mp.atan2(b,-a)
        def fun(t):
            nonlocal calls
            calls+=1;theta=angle*t
            return (a*mp.sin(theta)+b*mp.cos(theta))*mp.sin(theta)*angle
        value=mp.sqrt(2/mp.pi)*mp.quad(fun,[0,1]) if angle else mp.mpf(0)
        return {'value':mp.nstr(value,120),'evaluations':calls,'angle':mp.nstr(angle,120)}

def sector_reference(item,v):
    with mp.workdps(120):
        vv=[mpq(x) for x in v];calls=0
        if item['sector']=='full':
            return {'value':mp.nstr(mp.sqrt(sum(x*x for x in vv))/mp.sqrt(2*mp.pi),120),'evaluations':0}
        lo=[mpq(fhex(x)) for x in item['ray_lo']];hi=[mpq(fhex(x)) for x in item['ray_hi']]
        l=mp.atan2(lo[1],lo[0]);h=mp.atan2(hi[1],hi[0])
        if h<l:h+=2*mp.pi
        assert 0<=h-l<=mp.pi
        if h==l:return {'value':'0.0','evaluations':0}
        knots=[l,h]
        if vv[0] or vv[1]:
            root=mp.atan2(-vv[0],vv[1])
            knots+= [root+k*mp.pi for k in range(-3,4) if l<root+k*mp.pi<h]
        knots=sorted(set(knots));b=abs(vv[2])
        def fun(theta):
            nonlocal calls
            calls+=1;a=vv[0]*mp.cos(theta)+vv[1]*mp.sin(theta)
            if b==0:return max(a,mp.mpf(0))*mp.sqrt(mp.pi/2)
            return (b+a*(mp.pi/2+mp.atan(a/b)))/mp.sqrt(2*mp.pi)
        value=mp.quad(fun,knots)/(2*mp.pi)
        return {'value':mp.nstr(value,120),'evaluations':calls,'knots':[mp.nstr(x,120) for x in knots]}

def comparison(value,reference):
    with mp.workdps(220):
        x=mp.mpf(value);y=mp.mpf(reference);error=abs(x-y)
        tolerance=mp.mpf('1e-100') if y==0 else mp.mpf('1e-90')+abs(y)*mp.mpf('1e-80')
        return {'absolute_error':mp.nstr(error,120),'relative_error':None if y==0 else mp.nstr(error/abs(y),120),
                'tolerance':mp.nstr(tolerance,120),'pass':bool(error<=tolerance)}

def main():
    dest=HERE/'results.json';progress=HERE/'progress.jsonl';assert not dest.exists() and not progress.exists()
    p=json.loads((HERE/'protocol.json').read_text())
    for name,sha in p['pins'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha
    def record(x):
        with progress.open('a') as f:f.write(json.dumps(pack(x),allow_nan=False)+'\n')
    start=time.process_time();rows=[];out={'complete':False,'rows':rows,'source_protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()}
    def save():
        out.update(cpu=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(pack(out),indent=2,allow_nan=False)+'\n');temp.replace(dest)
    try:
        for item in p['scalar_controls']:
            record({'stage':'attempt','kind':'scalar','id':item['id']});t=time.process_time()
            a=fhex(item['a']);b=fhex(item['b']);c160=radial(a,b,160);c200=radial(a,b,200);reference=positive_scalar_reference(a,b)
            row={'id':item['id'],'kind':'scalar','inputs':item,'value160':c160,'value200':c200,'reference':reference,
                 'precision_check':comparison(c160,c200),'independent_check':comparison(c200,reference['value']),'cpu':time.process_time()-t}
            rows.append(row);record({'stage':'result','row':row});save()
        for item in p['facet_controls']:
            record({'stage':'attempt','kind':'facet','id':item['id']});t=time.process_time()
            normals=[[fhex(x) for x in n] for n in item['normals']];v=[fhex(x) for x in item['v']]
            certificate=prepare(normals,v);c160=evaluate(certificate,160);c200=evaluate(certificate,200);reference=sector_reference(item,v)
            row={'id':item['id'],'kind':'facet','inputs':item,'certificate':certificate,'value160':c160,'value200':c200,'reference':reference,
                 'precision_check':comparison(c160['value'],c200['value']),'independent_check':comparison(c200['value'],reference['value']),'cpu':time.process_time()-t}
            rows.append(row);record({'stage':'result','row':row});save()
        for item in p['invalid_controls']:
            record({'stage':'attempt','kind':'invalid','id':item['id']});error=None
            try:prepare([[F(x) for x in n] for n in item['normals']],[F(x) for x in item['v']])
            except (AssertionError,ValueError) as exc:error=type(exc).__name__
            row={'id':item['id'],'kind':'invalid','exception':error,'rejected':error is not None};rows.append(row);record({'stage':'result','row':row});save()
        out['complete']=True
        out['all_checks_pass']=all(r['rejected'] if r['kind']=='invalid' else r['precision_check']['pass'] and r['independent_check']['pass'] for r in rows)
    except BaseException as exc:
        out['failure']=repr(exc);record({'stage':'failure','exception':repr(exc)});raise
    finally:save()
    print(json.dumps({'complete':out['complete'],'all_checks_pass':out['all_checks_pass'],'rows':len(rows),'cpu':out['cpu']}))

if __name__=='__main__':main()
