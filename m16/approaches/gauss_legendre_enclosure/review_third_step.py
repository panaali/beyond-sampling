"""Saved T175 audit via canonical rays, exact Gram and acos facet formula."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,json,time
import mpmath as mp
H=Path(__file__).resolve().parent;A=Path('/workspace/work/recover-small_h2_34/m16/approaches/third_step_means')
def unpack(x):
    if isinstance(x,list) and len(x)==2 and all(isinstance(a,str) and a.lstrip('-').isdigit() for a in x):return F(int(x[0]),int(x[1]))
    if isinstance(x,list):return [unpack(a) for a in x]
    if isinstance(x,dict):return {k:unpack(v) for k,v in x.items()}
    return x
def dot(a,b):return sum((x*y for x,y in zip(a,b)),F(0))
def mq(x):return mp.mpf(x.numerator)/x.denominator
def close(a,b,tol='1e-180'):assert abs(mp.mpf(a)-b)<=mp.mpf(tol)*max(1,abs(b)),(a,str(b))
def main():
    out=H/'review_third_step.json';assert not out.exists();start=time.process_time();P=json.loads((H/'review_third_step_protocol.json').read_text())
    for f,h in P['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
    p=json.loads((A/'protocol.json').read_text());z=json.loads((A/'results.json').read_text())
    for f,h in p['pins'].items():assert hashlib.sha256((A/f).read_bytes()).hexdigest()==h
    assert z['complete'] and z['all_checks_pass'] and len(z['rows'])==35
    assert z['source_protocol_sha256']==hashlib.sha256((A/'protocol.json').read_bytes()).hexdigest()
    progress=[json.loads(l) for l in (A/'progress.jsonl').read_text().splitlines()];assert len(progress)==70
    total=Counter();statuses=Counter();evals=0;ids=set()
    with mp.workdps(250):
      for i,r in enumerate(z['rows']):
        key=(r['kind'],r['id']);assert key not in ids;ids.add(key)
        assert progress[2*i]==dict(stage='attempt',kind=r['kind'],id=r['id']) and progress[2*i+1]==dict(stage='result',row=r)
        if r['kind']=='invalid':assert r['rejected'] and r['exception'] in ['AssertionError','ValueError'];continue
        item=r['inputs'];v=None
        if r['kind']=='scalar':
            assert item in p['scalar_controls'];a=mq(F(float.fromhex(item['a'])));b=abs(mq(F(float.fromhex(item['b']))))
            if b==0:expected=max(a,0)*mp.sqrt(mp.pi/2)
            elif a<0:expected=(b-abs(a)*mp.atan(b/abs(a)))/mp.sqrt(2*mp.pi)
            else:expected=(b+a*(mp.pi/2+mp.atan(a/b)))/mp.sqrt(2*mp.pi)
            close(r['value200'],expected);x160=r['value160'];x200=r['value200']
        else:
            assert item in p['facet_controls'];v=[F(float.fromhex(x)) for x in item['v']];ns=[[F(float.fromhex(x)) for x in a] for a in item['normals']]+[v];c=unpack(r['certificate']);count=Counter();unique=[];canon=set();status='prepared'
            if not any(v):status='zero_output'
            else:
                for n in ns:
                    count.update(mul=3,add=2)
                    if not any(n):continue
                    k=next(abs(x) for x in n if x);key=tuple(x/k for x in n)
                    dup=False
                    for u in unique:
                        count.update(mul=8,add=4,sub=1)
                        uk=next(abs(x) for x in u if x);uk=tuple(x/uk for x in u)
                        if key==tuple(-x for x in uk):status='opposite_measure_zero';break
                        if key==uk:dup=True;break
                    if status!='prepared':break
                    if not dup:unique.append(n)
            assert status==c['status'];statuses[status]+=1
            if status!='prepared':expected=mp.mpf(0)
            else:
                assert unique==c['normals'];m=len(unique);count.update(mul=3*m*m+3*m,add=2*m*m+2*m);expected=mp.mpf(0)
                for j,(n,f) in enumerate(zip(unique,c['facets'])):
                    norm=dot(n,n);vn=dot(v,n);other=[u for k,u in enumerate(unique) if k!=j]
                    assert f['normal']==n and f['norm2']==norm and f['v_dot_normal']==vn and f['other_count']==len(other)
                    if not other:prob=mp.mpf(1)
                    elif len(other)==1:prob=mp.mpf('.5')
                    else:
                        u,w=other;p1=dot(u,u)*norm-dot(u,n)**2;q=dot(w,w)*norm-dot(w,n)**2;b=dot(u,w)*norm-dot(u,n)*dot(w,n);rad=p1*q-b*b
                        assert p1>0 and q>0 and rad>=0 and f['projected_norm_products']==[p1,q] and f['projected_dot']==b and f['angle_radicand']==rad
                        count.update(mul=8,sub=4)
                        prob=(mp.mpf(0) if b<0 else mp.mpf(".5")) if rad==0 else mp.acos(-mq(b)/mp.sqrt(mq(p1*q)))/(2*mp.pi)
                    term=mq(vn)*prob/mp.sqrt(2*mp.pi*mq(norm));expected+=term
                    close(r['value200']['terms'][j]['probability'],prob);close(r['value200']['terms'][j]['contribution'],term)
            assert dict(count)==c['counts'],(r['id'],count,c['counts']);total.update(count)
            close(r['value200']['value'],expected);x160=r['value160']['value'];x200=r['value200']['value']
        for name,x,y in [('precision_check',x160,x200),('independent_check',x200,r['reference']['value'])]:
            x=mp.mpf(x);y=mp.mpf(y);e=abs(x-y);tol=mp.mpf('1e-100') if y==0 else mp.mpf('1e-90')+abs(y)*mp.mpf('1e-80');s=r[name]
            close(s['absolute_error'],e,'1e-210');close(s['tolerance'],tol,'1e-110')
            if y:close(s['relative_error'],e/abs(y),'1e-210')
            else:assert s['relative_error'] is None
            assert s['pass']==(e<=tol)
        evals+=r['reference']['evaluations']
    result=dict(passed=True,cases=35,progress=70,facet_statuses=dict(statuses),exact_counts=dict(total),saved_reference_evaluations=evals,cpu_seconds=time.process_time()-start,scope='Independent saved domains/canonical proportionality/Gram/static counts and alternative acos facet contraction. Scalar negative-a alternative formula and saved reference/error ledgers checked; no author candidate or reference quadrature replay. MP arithmetic is diagnostic, not outward certified; no trained or downstream result.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        with (H/'review_third_step_failures.jsonl').open('a') as f:f.write(json.dumps(dict(type=type(e).__name__,message=str(e)))+'\n')
        raise
