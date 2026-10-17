"""Exact same-binary-coefficient diagnosis of frozen near-singular controls."""
import time
START=time.process_time()
from fractions import Fraction as F
import json,math
import numpy as np
import controls as Q
import kernel as K
out=dict(rows=[],scope='Exact closed-form equicorrelation diagnosis only; no trained panel or replacement kernel.')
for rho in [1-2**-20,1-2**-40]:
 s=np.array([1.,-1.,1.,-1.]);C=(1-rho)*np.eye(4)+rho*np.outer(s,s);coeff=K.coefficients(C.tolist());exactcoeff=K.coefficients([[F(float(v)) for v in row] for row in C]);worst=None
 for key,rules in Q.RULES.items():
  for idx,(t,w,jac) in enumerate(rules):
   pcs=K.correlations(coeff,float(t));r=F(float(t))*F(rho);truth=[float(s[k]*s[l])*float(r/(1+2*r)) for _,_,k,l in K.PAIRS]
   err=max(abs(a-b) for a,b in zip(pcs,truth))
   if worst is None or err>worst[0]:worst=(err,key,idx,float(t),pcs,truth)
 err,key,idx,t,pcs,truth=worst;v,p=K.polynomials(coeff,t);ve,pe=K.polynomials(exactcoeff,F(t));roundedcoeff=tuple([[F(float(x)) for x in row] for row in group] for group in coeff);vr,pr=K.polynomials(roundedcoeff,F(t));O=np.linalg.inv(np.eye(4)+t*(C-np.eye(4)));old=[-O[k,l]/math.sqrt(O[k,k]*O[l,l]) for _,_,k,l in K.PAIRS]
 row=dict(rho=rho,rule=key,node=idx,t=t,source_pc=old,candidate_pc=pcs,exact_pc=truth,max_source_error=max(abs(a-b) for a,b in zip(old,truth)),max_candidate_error=err,float_minors=v,exact_minors=[float(x) for x in ve],float_numerators=p,exact_numerators=[float(x) for x in pe],rounded_coefficient_exact_arithmetic_minors=[float(x) for x in vr],rounded_coefficient_exact_arithmetic_numerators=[float(x) for x in pr],exact_minor_positive=all(x>0 for x in ve),exact_identity_checks=[])
 for value,(_,_,k,l),vk,vl in zip(pe,K.PAIRS,[ve[k] for _,_,k,l in K.PAIRS],[ve[l] for _,_,k,l in K.PAIRS]):
  r=F(t)*F(rho);target=F(float(s[k]*s[l]))*r/(1+2*r);assert value*value==target*target*vk*vl;row['exact_identity_checks'].append(True)
 out['rows'].append(row)
out['cpu']=time.process_time()-START;(Q.HERE/'diagnosis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
