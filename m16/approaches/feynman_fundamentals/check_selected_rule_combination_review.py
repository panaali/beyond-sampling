"""Independent exact selected ideal-tail plus source-constant contraction."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,time
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/gauss_legendre_enclosure/selected_rules')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f(v):return F(int(v[0]),int(v[1]))
start=time.process_time();p=read(P/'combine_protocol.json');r=read(P/'combined.json')
assert r['complete'] and p['node_review'].split()[0]=='ed1fbdf5' and r['node_review']==p['node_review']
for path,h in p['pins'].items():assert sha(path)==h
old=read(p['old_constants']);split=read(p['split_bounds']);selection=read(p['selector'])
constants={6:next(c for c in old['constants'] if c['order']==6)};orders=[12,22,28,30]
assert [c['order'] for c in r['new_constants']]==orders
for n,saved in zip(orders,r['new_constants']):
 src=read(P/f'nodes_{n}.json');assert src['complete'];src=src['order'];assert src['n']==n and src['bounded']
 eps=delta=s=F();assert len(src['source_comparison'])==len(src['source_nodes_hex'])==len(src['source_weights_hex'])==len(saved['items'])==n
 for raw,xh,wh,it in zip(src['source_comparison'],src['source_nodes_hex'],src['source_weights_hex'],saved['items']):
  x=F((float.fromhex(xh)+1)/2);w=F(float.fromhex(wh)/2);lo,hi=[(f(v)+1)/2 for v in raw['ideal_node']];wl,wu=[f(v)/2 for v in raw['ideal_weight']]
  assert 0<=lo<=hi<=1 and 0<=x<=1 and 0<wl<=wu and w>=0
  dx=max(abs(x-lo),abs(x-hi));dw=max(abs(w-wl),abs(w-wu));eps=max(eps,dx);delta+=dw;s+=w
  assert (f(it['source_node']),f(it['source_weight']),f(it['node_error_upper']),f(it['weight_error_upper']))==(x,w,dx,dw)
  assert list(map(f,it['ideal_node']))==[lo,hi] and list(map(f,it['ideal_weight']))==[wl,wu]
 assert (f(saved['epsilon']),f(saved['delta_L1']),f(saved['source_weight_sum']))==(eps,delta,s) and abs(s-1)<=delta
 assert saved['source_mapping_arithmetic']==3*n and saved['counts']=={'rational_add':4*n,'rational_div':4*n,'rational_sub':4*n,'rational_abs':4*n,'extremum_comparison':3*n,'input_float_to_exact':2*n}
 constants[n]=saved
setup={(x['fixture'],x['side']):x for x in split['exact_setup']};selected={x['fixture']:x['selection'] for x in selection['groups'] if x['panel']=='child'}
assert {k:v['selected_order'] for k,v in selected.items()}==p['fixed_selection'] and len(selected)==11
values={};reused=0
for row in r['rows']:
 label,side,n=row['fixture'],row['side'],row['order'];assert n==selected[label]['selected_order'];key=(label,side);ss=setup[key];a,M,d=map(f,(ss['a'],ss['M2'],ss['det']));cn=constants[n];eps,s,delta=map(f,(cn['epsilon'],cn['source_weight_sum'],cn['delta_L1']));g=row['result'];assert g['status']=='bounded'
 if n==6:
  prior=next(x for x in split['rows'] if (x['fixture'],x['side'],x['order'])==(label,side,n));assert g==prior['result'] and row['origin']=='reused';reused+=1
 else:assert row['origin']=='new'
 k=math.isqrt((a.numerator<<320)//a.denominator);lo=F(k,1<<160);assert lo>0 and lo*lo<=a<F(k+1,1<<160)**2
 q5=1/(a*a*lo);q7=q5/a;L=6*q5+40*M*q7;node=s**4*eps*L;weight=delta*(s**3+s*s+s+1)*q5;v=abs(d)*(node+weight)/24
 for name,q in [('q5_upper',q5),('q7_upper',q7),('derivative_sum_upper',L),('node_upper',node),('weight_upper',weight),('integral_upper',node+weight),('prefactor_upper',abs(d)/24),('probability_upper',v)]:assert f(g[name])==q
 assert g['counts']=={'comparison_lt':6,'integer_shift':2,'integer_divide':1,'integer_isqrt':1,'integer_add':1,'rational_mul':13,'rational_div':3,'rational_add':5,'rational_abs':1}
 assert key not in values;values[key]=v
assert len(values)==22 and reused==r['reused_child_bounds']==4 and r['new_child_bounds']==18
maxbound=maxconstant=F();margin=F(1);outrows=[]
for g in r['groups']:
 label=g['fixture'];sel=selected[label];ideal=F(sel['bound_sum']);assert sel['status']=='certified_ideal' and ideal==F(sel['steps'][-1]['bound_sum'])
 constant=values[label,0]+values[label,1];total=ideal+constant
 assert g['order']==sel['selected_order'] and g['status']=='bounded' and (f(g['ideal_upper']),f(g['constant_upper']),f(g['combined_upper']))==(ideal,constant,total)
 assert g['below_1e_10']==(total<=F(1,10**10)) and total<=F(1,10**10)
 maxbound=max(maxbound,total);maxconstant=max(maxconstant,constant);margin=min(margin,F(1,10**10)-total)
 outrows.append(dict(fixture=label,order=g['order'],ideal=str(ideal),constant=str(constant),combined=str(total)))
assert len(outrows)==11 and len({g['fixture'] for g in outrows})==11
assert r['source_mapping_arithmetic']==3*sum(orders)==276 and r['exact_parent_assembly_additions_allowance']==33 and r['new_certificate_determinant_calls']==r['new_source_leggauss_calls']==r['tensor_mass_model_geometry_MC_calls']==0
progress=[json.loads(x) for x in (P/'combined_progress.jsonl').read_text().splitlines()];assert len(progress)==46
out=dict(complete=True,parents=11,child_bounds=22,new_bounds=18,reused_bounds=4,all_below_target=True,max_combined=float(maxbound),max_constant=float(maxconstant),minimum_target_margin=float(margin),rows=outrows,combined_sha256=sha(P/'combined.json'),protocol_sha256=sha(P/'combine_protocol.json'),node_review=p['node_review'],review_cpu=time.process_time()-start,new_mass_model_or_tensor_calls=0,scope='Exact mapped constants, sensitivity expressions, prior fixed selections and combined rational sums verified. Node certificates independently checked by34. Ideal-real quadrature with source constants only; actual map/kernel/determinant/accumulation errors and construction/advice costs remain separate.')
Path(__file__).with_name('selected_rule_combination_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
