"""Frozen synthetic gate, tie and trained-parity controls."""
import time
START=time.process_time()
import hashlib,inspect,json,sys,types,traceback,resource
from pathlib import Path
from fractions import Fraction as F
import numpy as np
P=Path(__file__).resolve().parent

def native(name):
    s=(P/'_source/geometry_native.py').read_text().replace("'data/dd_geometry.so'",repr('data/'+name+'.so'))
    m=types.ModuleType('geometry_'+name);m.__file__=str(P/'wrapper.py');exec(compile(s,m.__file__,'exec'),m.__dict__);return m
NEW=native('fixed');OLD=native('original');sys.modules['geometry_native']=NEW
REF=types.ModuleType('reference');REF.__file__=str(P/'reference.py')
exec(compile((P/'_source/check.py').read_text(),str(P/'_source/check.py'),'exec'),REF.__dict__)
E=REF.E;s=inspect.getsource(E._children)
anchor='        new = []\n        for R1, Z1, g in pieces:'
assert s.count(anchor)==1
s=s.replace(anchor,'''        if not np.any(a):
            pieces=[(R1,Z1|bit(cid(k)),g) for R1,Z1,g in pieces]
            ops.add(len(a)+sum(len(R1) for R1,_,_ in pieces), "exactzero_gate")
            continue
'''+anchor)
exec(compile(s,str(P/'python_zero_reference.py'),'exec'),E.__dict__)

def map_checks(model,data):
    wi,wh,wo=model;ptr=0;mi=0;maxerr=0.;exactcount=0
    for typ,n in data['headers']:
        R=data['rays'][ptr:ptr+n];ptr+=n
        if typ!=0:continue
        A=data['maps'][mi];mi+=1;x=R.sum(0);h=np.zeros(2)
        for value in x:h=np.maximum(wh@h+wi*value,0)
        maxerr=max(maxerr,float(np.max(abs(A@x-h))))
        # Fractions independently verify every fixed-map value on this interior.
        W=np.vectorize(lambda v:F(float(v)))(wh);I=np.array([F(float(v)) for v in wi],object);hh=np.array([F(0),F(0)],object)
        xx=np.array([F(float(v)) for v in x],object)
        for value in xx:hh=np.array([max(v,F(0)) for v in W@hh+I*value],object)
        AA=np.vectorize(lambda v:F(float(v)))(A);error=max(abs(float(v)) for v in AA@xx-hh)
        assert error<1e-10,('exact map mismatch',error);exactcount+=1
    assert mi==len(data['maps']) and ptr==len(data['rays'])
    return dict(exact_interior_checks=exactcount,max_double_map_error=maxerr)

def main():
    wi=np.array([1.,0.]);wo=np.zeros((5,2));wo[4,0]=1;zero=np.zeros((2,2))
    cases=[('zero_second',(wi,zero,wo),89/320),('permuted',(wi[::-1],zero,wo[:,::-1]),89/320),
        ('zero_readout',(wi,zero,np.zeros((5,2))),.2),('duplicate_readout',(wi,zero,np.tile([1.,0.],(5,1))),.2),
        ('conditional_zero',(wi,np.array([[0.,0.],[1.,0.]]),np.array([[0.,0.],[0.,1.],[0.,0.],[0.,0.],[1.,0.]])),None)]
    for scale in [1e-6,1e6]:cases.append((f'scale{scale}',(wi*scale,zero,wo/scale),89/320))
    rows=[];out=dict(complete=False,cases=rows,source_pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [REF.SOURCE,P/'fixed.cpp',P/'original.cpp',P/'check.py']})
    for name,model,truth in cases:
        row=dict(name=name,analytic_accuracy=truth,weights=[w.tolist() for w in model]);tick=time.process_time()
        try:OLD.geometry(model)
        except RuntimeError as e:row['original_rejection']=str(e)
        else:row['original_rejection']=None
        old=REF.reference(model);new=NEW.geometry(model)
        temp=dict(new);temp['counts']=new['counts'][:13]
        row['reference_difference']=REF.compare(old,temp,model);row['counts']=new['counts'].tolist();row['validation']=map_checks(model,new)
        row['cpu_seconds']=time.process_time()-tick;rows.append(row)
        out['cpu_seconds_including_imports']=time.process_time()-START;(P/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
        print(name,new['counts'].tolist(),flush=True)
        assert out['cpu_seconds_including_imports']<95
    # Nonzero small coefficient must retain original path and all original arrays.
    model=(np.array([1.,1e-8]),zero,wo);a=OLD.geometry(model);b=NEW.geometry(model);tmp=dict(b);tmp['counts']=b['counts'][:13]
    diff=REF.compare(a,tmp,model);assert b['counts'][13]==0
    out['nearzero_control']=dict(difference=diff,counts=b['counts'].tolist())
    rejected=[]
    for name,model in [('allzero_input',(np.zeros(2),np.eye(2),wo)),('nonfinite',(np.array([np.nan,0]),zero,wo))]:
        try:NEW.geometry(model)
        except (ValueError,RuntimeError) as e:rejected.append(dict(name=name,error=str(e)))
        else:raise AssertionError('unsupported input accepted')
    out.update(complete=True,rejections=rejected,cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (P/'controls.json').write_text(json.dumps(out,indent=2)+'\n');print('CPU',out['cpu_seconds_including_imports'])
if __name__=='__main__':
    try:main()
    except Exception as e:
        (P/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),cpu_seconds=time.process_time()-START),indent=2)+'\n');raise
