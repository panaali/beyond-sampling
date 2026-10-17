"""Exact rational three-suffix partition on signed L1 simplex faces."""
import time
START=time.process_time()
import hashlib,json,resource
from fractions import Fraction as F
from pathlib import Path
P=Path(__file__).resolve().parent
SOURCE=Path('/workspace/work/recover-small_h2_34/m16/approaches/reset_wedge/results.json')
COUNTS=dict(clips=0,plane_evaluations=0,intersections=0,direct_checks=0)
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def area(poly):return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(poly,poly[1:]+poly[:1])))/2 if poly else F(0)
def clip(poly,normal):
    COUNTS['clips']+=1
    if not poly:return []
    out=[];vals=[dot(normal,v) for v in poly];COUNTS['plane_evaluations']+=len(poly)
    for a,b,da,db in zip(poly,poly[1:]+poly[:1],vals,vals[1:]+vals[:1]):
        if da>=0:out.append(a)
        if (da<0<db) or (db<0<da):
            ratio=da/(da-db);out.append(tuple(x+ratio*(y-x) for x,y in zip(a,b)));COUNTS['intersections']+=1
    out=list(dict.fromkeys(out))
    return out if len(out)>=3 and area(out)>0 else []
def transform(normal,sign):return tuple(x*s for x,s in zip(normal,sign))
def partition(wi,wh,wo,k):
    initial=[(F(1),F(0),F(0)),(F(0),F(1),F(0)),(F(0),F(0),F(1))];rows=[];stats=[]
    for sign in ((1,1,1),(1,1,-1),(1,-1,1)):
        domain=initial[:]
        if sign[1]<0:domain=clip(domain,(k,-1,0))
        if sign[2]<0:domain=clip(domain,(0,k,-1))
        cells=[(domain,((F(0),)*3,)*2,())]
        for t in range(3):
            nextcells=[]
            for poly,A,history in cells:
                B=tuple(tuple(sum(wh[i][j]*A[j][q] for j in range(2))+ (wi[i] if q==t else 0) for q in range(3)) for i in range(2))
                pieces=[(poly,())]
                for i in range(2):
                    split=[];normal=transform(B[i],sign)
                    for poly,gates in pieces:
                        if not any(normal):split.append((poly,gates+(False,)));continue
                        for on in (False,True):
                            clipped=clip(poly,normal if on else tuple(-v for v in normal))
                            if clipped:split.append((clipped,gates+(on,)))
                    pieces=split
                for poly,gates in pieces:nextcells.append((poly,tuple(B[i] if gates[i] else (F(0),)*3 for i in range(2)),history+(gates,)))
            cells=nextcells
            assert len(cells)<=256 and time.process_time()-START<30
        assert sum(area(c[0]) for c in cells)==area(domain)
        n0=len(rows)
        for poly,A,history in cells:
            logits=tuple(tuple(sum(row[i]*A[i][q] for i in range(2)) for q in range(3)) for row in wo)
            winnerareas=[]
            for label in range(len(wo)):
                region=poly
                for other in range(len(wo)):
                    if other==label:continue
                    dl=tuple(a-b for a,b in zip(logits[label],logits[other]))
                    if not any(dl):
                        if other<label:region=[];break
                    else:region=clip(region,transform(dl,sign))
                    if not region:break
                if not region:continue
                winnerareas.append(area(region))
                for offset in (0,1,2):
                    weights=[F(1+(j+offset)%3) for j in range(len(region))];den=sum(weights)
                    v=tuple(sum(w*p[q] for w,p in zip(weights,region))/den for q in range(3));x=transform(v,sign);h=(F(0),F(0))
                    for z in x:h=tuple(max(F(0),dot(row,h)+wi[i]*z) for i,row in enumerate(wh))
                    assert h==tuple(dot(row,x) for row in A)
                    direct=tuple(dot(row,h) for row in wo);assert max(range(len(wo)),key=lambda j:direct[j])==label
                    assert x[0]>0 and x[1]>-k*x[0] and x[2]>-k*max(x[1],F(0));COUNTS['direct_checks']+=1
                rows.append(dict(sign=sign,vertices=region,A=A,history=history,label=label))
            assert sum(winnerareas)==area(poly)
        stats.append(dict(sign=sign,domain_area=str(area(domain)),gate_cells=len(cells),winner_cells=len(rows)-n0))
    assert len(rows)<=1024
    return rows,stats

def main():
    saved=json.loads(SOURCE.read_text());model=saved['models'][0]
    wi,wh,wo=model['weights'];wi=tuple(F(float(x)) for x in wi);wh=tuple(tuple(F(float(x)) for x in row) for row in wh);wo=tuple(tuple(F(float(x)) for x in row) for row in wo)
    a,b,c,d=-wh[0][0],-wh[0][1],wh[1][0],-wh[1][1];assert min(wi)>0 and min(a,b,c,d)>=0
    gamma=c*wi[0] if d==0 else max(F(0),c*wi[0]-d*wi[1],c*max(wi[0]-b*wi[1]/d,F(0)));k=gamma/wi[1]
    rows,stats=partition(wi,wh,wo,k);T=len(wo);labels=sorted(set(r['label'] for r in rows));unsupported=[i for i,r in enumerate(rows) if r['label']<T-3 or r['sign'][r['label']-(T-3)]<0]
    bits=max(max(abs(x.numerator).bit_length(),x.denominator.bit_length()) for r in rows for v in r['vertices'] for x in v)
    result=dict(complete=True,model_sha256=model['model_sha256'],source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),source_path=str(SOURCE),kappa=str(k),orthants=stats,labels=labels,unsupported_positive_suffix_cells=unsupported,max_vertex_integer_bits=bits,counts=COUNTS,cells=rows,cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (P/'topology.json').write_text(json.dumps(result,indent=2,default=str)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))
if __name__=='__main__':main()
