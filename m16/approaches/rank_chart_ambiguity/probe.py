"""Finite deterministic exact open-witness test; absent witness is unknown."""
import time
START=time.process_time()
import gzip,hashlib,itertools,json,math,resource
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=Path('/workspace/clean-neurons/m16/approaches/small_ladder_review/rational_geometry/geometry_h4_t4.json.gz')
COUNTS=Counter()
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def primitive(row):
 g=math.gcd(*row)
 return tuple(v//g for v in row) if g else tuple(row)
def inverse(rows):
 n=len(rows);a=[[F(v) for v in row]+[F(i==j) for j in range(n)] for i,row in enumerate(rows)]
 for j in range(n):
  k=next(k for k in range(j,n) if a[k][j]);a[j],a[k]=a[k],a[j];v=a[j][j];a[j]=[x/v for x in a[j]]
  for k in range(n):
   if k!=j:
    v=a[k][j];a[k]=[x-v*y for x,y in zip(a[k],a[j])]
 out=[r[n:] for r in a]
 assert all(sum(rows[i][k]*out[k][j] for k in range(n))==int(i==j) for i in range(n) for j in range(n))
 return out

def chart(a,b,sign):
 rows=[tuple(int(q==i)-int(q==j) for q in range(4)) for i,j in [(a,b)]+[(b,j) for j in range(4) if j not in (a,b)]]+[tuple(sign*int(q==0) for q in range(4))]
 positive=F(15 if a==0 else 11 if b==0 else 3,16)
 mass=(positive if sign>0 else 1-positive)/12
 return rows,inverse(rows),mass

def recurrence(x,wi,wh,wo,D,maps=True):
 H=len(wi);h=(0,)*H;A=((0,)*4,)*H;scale=1;history='';planes=[];boundary=False
 for t in range(4):
  values=tuple(dot(row,h)+wi[i]*x[t]*scale for i,row in enumerate(wh));on=tuple(v>0 for v in values);history+=''.join('1' if v else '0' for v in on)
  if maps:
   B=tuple(tuple(sum(wh[i][j]*A[j][q] for j in range(H))+(wi[i]*scale if q==t else 0) for q in range(4)) for i in range(H));assert tuple(dot(row,x) for row in B)==values
   for row,active in zip(B,on):
    if any(row):
     row=primitive(row if active else tuple(-v for v in row));planes.append(row)
     if dot(row,x)==0:boundary=True
    else:assert not active
   A=tuple(row if active else (0,)*4 for row,active in zip(B,on))
  h=tuple(max(v,0) for v in values);scale*=D
 logits=tuple(dot(row,h) for row in wo);prediction=max(range(4),key=lambda i:logits[i]);COUNTS['direct_network_evaluations']+=1
 if maps:
  L=tuple(tuple(sum(row[i]*A[i][q] for i in range(H)) for q in range(4)) for row in wo);assert tuple(dot(row,x) for row in L)==logits;COUNTS['direct_vs_mapped_logit_checks']+=1
 else:L=None
 return prediction,history,planes,L,boundary

def witness(x,rows,target,model):
 prediction,history,gates,L,boundary=recurrence(x,*model)
 kind='correct' if prediction==target else 'wrong';constraints=rows+gates;competitor=None
 if boundary:return kind,None,'gate_boundary'
 if kind=='correct':
  for j in range(4):
   if j==target:continue
   margin=primitive(tuple(a-b for a,b in zip(L[target],L[j])))
   if any(margin):
    if dot(margin,x)==0:return kind,None,'output_boundary'
    assert dot(margin,x)>0;constraints.append(margin)
   else:assert j>target
 else:
  for j in range(4):
   if j==target:continue
   margin=primitive(tuple(a-b for a,b in zip(L[j],L[target])))
   if dot(margin,x)>0:constraints.append(margin);competitor=j;break
   if not any(margin) and j<target:competitor=j;break
  if competitor is None:return kind,None,'output_boundary'
 assert all(dot(row,x)>0 for row in constraints)
 radius=min(F(dot(row,x),2*sum(abs(v) for v in row)) for row in constraints);assert radius>0
 assert all(F(dot(row,x))-radius*sum(abs(v) for v in row)>0 for row in constraints)
 COUNTS['open_neighborhood_certificates']+=1
 return kind,dict(x=list(map(str,x)),history=history,prediction=prediction,kind=kind,competitor=competitor,radius_linf=str(radius),constraint_count=len(constraints)),None

def main():
 stored=SOURCE.read_bytes();data=json.loads(gzip.decompress(stored));decode=lambda p:F(int(p[0]),int(p[1]));wi=list(map(decode,data['model'][0]));wh=[list(map(decode,row)) for row in data['model'][1]];wo=[list(map(decode,row)) for row in data['model'][2]];D=max(v.denominator for v in wi+sum(wh,[])+sum(wo,[]));assert D&(D-1)==0
 cv=lambda row:tuple(int(v*D) for v in row);model=(cv(wi),tuple(map(cv,wh)),tuple(map(cv,wo)),D)
 charts=[]
 for sign in (1,-1):
  for a,b in itertools.permutations(range(4),2):
   rows,inv,mass=chart(a,b,sign);charts.append((a,b,sign,rows,inv,mass))
 assert sum(c[-1] for c in charts)==1
 fixtures=[('constant_first',((0,),((0,),),((0,),)*4,1),0),('constant_last',((1,-1),((1,0),(0,1)),((0,0),(0,0),(0,0),(1,1)),1),3)]
 fixture_results=[]
 for name,fixture,expected in fixtures:
  for a,b,sign,rows,inv,mass in charts:
   x=tuple(dot(row,(1,4,16,1)) for row in inv);assert all(v.denominator==1 for v in x);x=tuple(map(int,x));pred,*_=recurrence(x,*fixture,maps=False);assert pred==expected
  fixture_results.append(dict(name=name,chart_checks=24,expected_prediction=expected,exact_accuracy='1/4'))
 out=dict(complete=False,source_sha256=hashlib.sha256(stored).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checkpoint_sha256=data['checkpoint_sha256'],fixtures=fixture_results,charts=[],random_inputs=0)
 for a,b,sign,rows,inv,mass in charts:
  row=dict(top=a,second=b,sign_x0=sign,analytic_mass=str(mass),outcomes={},rejections={},witnesses={});outcomes=Counter();rejections=Counter()
  for coefficients in itertools.product((1,4,16),repeat=4):
   assert time.process_time()-START<20 and COUNTS['direct_network_evaluations']<2000
   x=tuple(dot(r,coefficients) for r in inv);assert all(v.denominator==1 for v in x);x=tuple(map(int,x));assert all(dot(n,x)==c for n,c in zip(rows,coefficients))
   kind,cert,rejected=witness(x,rows,b,model);outcomes[kind]+=1
   if rejected:rejections[rejected]+=1
   elif kind not in row['witnesses']:row['witnesses'][kind]=dict(cert,coefficients=list(coefficients))
  row.update(outcomes=dict(outcomes),rejections=dict(rejections),status='mixed_open' if len(row['witnesses'])==2 else 'unknown');out['charts'].append(row)
 out.update(complete=True,counts=dict(COUNTS),mixed_charts=sum(r['status']=='mixed_open' for r in out['charts']),mixed_chart_mass=str(sum(F(r['analytic_mass']) for r in out['charts'] if r['status']=='mixed_open')),cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 out['whole_chart_uniform_coverage_ceiling']=str(1-F(out['mixed_chart_mass']));(HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('charts','fixtures')}))
if __name__=='__main__':main()
