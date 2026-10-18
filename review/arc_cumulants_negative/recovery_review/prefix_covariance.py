"""Exact-real angular-sector first/two moments; ordinary binary64 implementation."""
import math
import numpy as np

def prefix2(wi,wh):
    wi=np.asarray(wi,float);wh=np.asarray(wh,float);h=len(wi)
    mu=np.zeros(h);raw=np.zeros((h,h));sectors=[]
    for sign in (1,-1):
        a=wh@np.maximum(sign*wi,0);B=np.column_stack((a,wi));cuts=[-math.pi/2,math.pi/2]
        for aa,bb in B:
            if aa==0 and bb==0:continue
            phi=math.atan2(float(bb),float(aa))
            for k in range(-2,3):
                z=phi+math.pi/2+k*math.pi
                if -math.pi/2<z<math.pi/2:cuts.append(z)
        cuts=sorted(set(cuts))
        for lo,hi in zip(cuts,cuts[1:]):
            mid=(lo+hi)/2;d=(hi-lo)/2;c=math.cos(mid);s=math.sin(mid)
            active=B@np.array([c,s])>0;A=B*active[:,None]
            first=2*math.sin(d)*np.array([c,s])
            # Integral sin(v)^2 over [-d,d], stable for narrow sectors.
            if d<.001:
                perpendicular=d**3*(2/3+d*d*(-2/15+d*d*(4/315+d*d*(-2/2835))))
            else:perpendicular=d-.5*math.sin(2*d)
            parallel=2*d-perpendicular;R=np.array([[c,-s],[s,c]])
            J=(R*np.array([parallel,perpendicular]))@R.T
            mu+=math.sqrt(math.pi/2)/(2*math.pi)*(A@first)
            raw+=(A@J@A.T)/math.pi
            sectors.append(dict(sign=sign,lo=lo,hi=hi,active=int(active.sum()),thin_series=bool(d<.001)))
    cov=raw-np.outer(mu,mu)
    return mu,cov,dict(sectors=sectors,sector_count=len(sectors),H=h,count_scope='Operation structure only, not complete scalar cost: two dense HxH matvecs; per sector Hx2 gate/mean products and Hx2 times2x2 times2xH second-moment products, trigonometry, root sorting and setup. Exact-reference work separate.')
