"""Independent single trained saved-sector/facet audit; no author imports or quadrature."""
from fractions import Fraction as F
from decimal import Decimal,localcontext
from pathlib import Path
import hashlib,json,time
import mpmath as mp
HERE=Path(__file__).resolve().parent

def frac(x):return F(int(x[0]),int(x[1]))
def vec(x):return tuple(map(frac,x))
def mat(x):return [vec(r) for r in x]
def dot(x,y):return sum((a*b for a,b in zip(x,y)),F(0))
def q(x):return mp.mpf(x.numerator)/x.denominator

def main():
 out=HERE/'trained_results.json';assert not out.exists();start=time.process_time()
 pin=json.loads((HERE/'trained_pins.json').read_text())
 for path,sha in pin.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
 path=Path('/workspace/work/recover-small_h2_34/m16/approaches/third_step_means/trained/result.json')
 raw=json.loads(path.read_text());geom=json.loads(path.with_name('geometry.json').read_text());assert raw['complete'] and raw['all_checks_pass'] and geom['complete']
 source_protocol=json.loads(path.with_name('protocol.json').read_text());assert raw['protocol_sha256']==pin[str(path.with_name('protocol.json'))]
 weights=json.loads(Path(source_protocol['weights_json']).read_text());assert weights['complete'] and len(weights['rows'])==1
 model=weights['rows'][0];assert model['name']==raw['model']=='H16T4seed3' and model['passed']
 saved={'name':raw['model'],'complete':True,'all_checks_pass':True,'certificate':geom['certificate'],'map_controls':geom['map_controls'],'cells':raw['rows']}
 item={'name':raw['model'],'wi_hex':[float(x).hex() for x in model['wi']],'wh_hex':[[float(x).hex() for x in row] for row in model['wh']]}
 d={'rows':[saved]};protocol={'controls':[item]}
 counts={'networks':0,'sectors':0,'cells':0,'map_records':0,'reference_evaluations':0};maximum=mp.mpf(0);rows=[]
 mp.mp.dps=110
 for saved,item in zip(d['rows'],protocol['controls'],strict=True):
  assert saved['name']==item['name'] and saved['complete'] and saved['all_checks_pass'];counts['networks']+=1
  c=saved['certificate'];wi=vec(c['wi']);wh=mat(c['wh']);h=len(wi);sectors=c['sectors'];S=len(sectors)
  assert wi==tuple(F(float.fromhex(x)) for x in item['wi_hex'])
  assert wh==[tuple(F(float.fromhex(x)) for x in r) for r in item['wh_hex']]
  for branch in c['branches']:
   sign=branch['sign'];first=tuple(max(F(0),sign*w) for w in wi);a=tuple(dot(r,first) for r in wh);assert vec(branch['a'])==a
   roots=sorted(set(-aa/b for aa,b in zip(a,wi) if b));assert list(map(frac,branch['root_slopes']))==roots
   groups=[[j for j,(aa,b) in enumerate(zip(a,wi)) if b and -aa/b==root] for root in roots];assert groups==branch['coincident_root_groups']
   cuts=[(F(0),F(-1))]+[(F(1),r) for r in roots]+[(F(0),F(1))]
   ss=[s for s in sectors if s['sign']==sign];assert len(ss)==len(cuts)-1
   for idx,s in enumerate(ss):
    lo,hi=vec(s['lo']),vec(s['hi']);assert (lo,hi)==(cuts[idx],cuts[idx+1]) and s['index']==idx
    rep=vec(s['representative']);assert rep[0]>0
    normals=[(-lo[1],lo[0],F(0)),(hi[1],-hi[0],F(0))];assert mat(s['normals'])==normals
    B=mat(s['h2_matrix']);assert len(B)==h
    for j,(aa,b) in enumerate(zip(a,wi)):
     on=dot((aa,b),rep)>0;assert s['active'][j]==on and B[j]==((aa,b) if on else (F(0),F(0)))
     vals=[dot((aa,b),r) for r in (lo,hi)]
     assert all(v>=0 for v in vals) if on else all(v<=0 for v in vals)
    V=mat(s['h3_rows']);assert V==[(sum((wh[j][k]*B[k][0] for k in range(h)),F(0)),sum((wh[j][k]*B[k][1] for k in range(h)),F(0)),wi[j]) for j in range(h)]
  assert c['exact_named_counts']=={'mul':2*h*h+S*(2*h+2*h*h),'add':2*h*(h-1)+S*(2+h+2*h*(h-1))}
  ex=c['extra_named_counts'];assert ex['input_sign_multiplications']==ex['input_positive_part_comparisons']==2*h and ex['activation_comparisons']==S*h and ex['boundary_normal_negations']==2*S
  assert ex.get('exact_slope_divisions',0)==ex.get('exact_slope_negations',0)==2*sum(w!=0 for w in wi)
  maps=saved['map_controls'];assert maps['direct_analytic_forwards']==len(maps['rows'])==15*S
  assert maps['exact_named_counts']=={'mul':5*S*(4*h*h+16*h),'add':5*S*(4*h*h+7*h)}
  for m in maps['rows']:
   sec=next(s for s in sectors if s['sign']==m['sign'] and s['index']==m['sector']);pt=vec(m['point']);assert m['pass'] and all(dot(n,(*pt,F(0)))>=0 for n in mat(sec['normals']))
   assert frac(m['z']) in (-1,0,1)
  assert len(saved['cells'])==S*h
  means=[mp.mpf(0) for _ in range(h)];refs=[mp.mpf(0) for _ in range(h)]
  for cell in saved['cells']:
   sec=sectors[cell['sector']];unit=cell['unit'];fc=cell['facet'];v=vec(fc['v']);assert v==mat(sec['h3_rows'])[unit]
   original=mat(sec['normals'])+[v];keys=[];reduced=[]
   for normal in original:
    first=next((x for x in normal if x),None)
    if first is None:continue
    key=tuple(x/abs(first) for x in normal)
    if key not in keys:keys.append(key);reduced.append(normal)
   opposite=any(tuple(-x for x in key) in keys for key in keys)
   expected='zero_output' if not any(v) else ('opposite_measure_zero' if opposite else 'prepared')
   assert fc['status']==expected
   if expected=='prepared':assert mat(fc['normals'])==reduced
   total=mp.mpf(0)
   if fc['status']=='prepared':
    normals=mat(fc['normals'])
    for i,f in enumerate(fc['facets']):
     n=vec(f['normal']);assert n==normals[i];nn=dot(n,n);vn=dot(v,n);assert frac(f['norm2'])==nn and frac(f['v_dot_normal'])==vn
     others=[u for j,u in enumerate(normals) if j!=i];assert len(others)==f['other_count']
     if len(others)==0:prob=mp.mpf(1)
     elif len(others)==1:prob=mp.mpf('.5')
     else:
      u,w=others;p=dot(u,u)*nn-dot(n,u)**2;r=dot(w,w)*nn-dot(n,w)**2;b=dot(u,w)*nn-dot(n,u)*dot(n,w)
      assert [frac(x) for x in f['projected_norm_products']]==[p,r] and frac(f['projected_dot'])==b and frac(f['angle_radicand'])==p*r-b*b
      rho=mp.mpf(1 if b>0 else -1) if b*b==p*r else q(b)/mp.sqrt(q(p*r))
      prob=mp.mpf('.25')+mp.asin(rho)/(2*mp.pi)
     total+=q(vn)/mp.sqrt(q(nn))*prob/mp.sqrt(2*mp.pi)
   else:assert fc['status'] in ('zero_output','opposite_measure_zero')
   target=mp.mpf(cell['value200']['value']);err=abs(total-target);maximum=max(maximum,err);assert err<=mp.mpf('1e-90')*(1+abs(target))
   reference=mp.mpf(cell['reference']['value']);assert abs(target-reference)<=mp.mpf('1e-60')+abs(reference)*mp.mpf('1e-50')
   assert abs(target-mp.mpf(cell['value160']['value']))<=mp.mpf('1e-100')*(1+abs(target))
   means[unit]+=target;refs[unit]+=reference;counts['reference_evaluations']+=cell['reference']['evaluations']
  counts['sectors']+=S;counts['cells']+=len(saved['cells']);counts['map_records']+=len(maps['rows'])
  rows.append({'name':saved['name'],'sectors':S,'cells':len(saved['cells']),'maps':len(maps['rows']),'passed':True})
 assert counts=={'networks':1,'sectors':34,'cells':544,'map_records':510,'reference_evaluations':222042}
 result={'complete':True,'counts':counts,'rows':rows,'max_independent_facet_difference':mp.nstr(maximum,90),'cpu_seconds':time.process_time()-start,'pins':pin,
  'author_or_quadrature_calls':0,'scope':'Independent exact stored-weight sector/conic gate-map algebra, finite saved coverage and alternate asin cell contraction for one H16T4seed3 model. Saved quadrature only compared, not rerun. Mean/baseline/reference aggregate review belongs to T55; no covariance/final-accuracy/outward/fullcost inference.'}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['complete','counts','max_independent_facet_difference','cpu_seconds']}))
if __name__=='__main__':main()
