"""Independent equicorrelation identity for preserved finite kernel failures."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from fractions import Fraction as F
from pathlib import Path
import importlib.util,json,time
start=time.process_time();H=Path(__file__).parent;P=Path('/workspace/work/recover-small_h2_910/m16/approaches/schur_polynomial_kernel');spec=importlib.util.spec_from_file_location('failed_kernel',P/'kernel.py');K=importlib.util.module_from_spec(spec);spec.loader.exec_module(K);D=json.loads((P/'diagnosis.json').read_text());out=[];sign=[1,-1,1,-1]
for row in D['rows']:
 rho=F(row['rho']);t=F(row['t']);r=rho*t;minor=(1-r)**2*(1+2*r);numerators=[sign[k]*sign[l]*r*(1-r)**2 for _,_,k,l in K.PAIRS];pc=[float(sign[k]*sign[l]*r/(1+2*r)) for _,_,k,l in K.PAIRS]
 assert pc==row['exact_pc'] and [float(minor)]*4==row['exact_minors'] and list(map(float,numerators))==row['exact_numerators']
 C=[[float(1 if i==j else sign[i]*sign[j]*rho) for j in range(4)] for i in range(4)];co=K.coefficients(C);v,p=K.polynomials(co,float(t));assert v==row['float_minors'] and p==row['float_numerators'];rounded=tuple([tuple(F(v) for v in vals) for vals in group] for group in co);vr,pr=K.polynomials(rounded,t);assert list(map(float,vr))==row['rounded_coefficient_exact_arithmetic_minors'] and list(map(float,pr))==row['rounded_coefficient_exact_arithmetic_numerators']
 err=max(abs(a-b) for a,b in zip(row['candidate_pc'],pc));assert err==row['max_candidate_error'];out.append(dict(rho=float(rho),t=float(t),candidate_correlation_error=err,source_correlation_error=row['max_source_error'],exact_numerator=float(numerators[0]),rounded_coefficient_exact_numerator=float(pr[0]),float_numerator=p[0]))
result=dict(cases=out,review_cpu=time.process_time()-start,scope='Two preserved frozen synthetic failures independently confirmed by closed-form signed equicorrelation law. Both coefficient construction and evaluation matter; no trained panel ran and no method-family impossibility follows.')
(H/'schur_failure_review_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
