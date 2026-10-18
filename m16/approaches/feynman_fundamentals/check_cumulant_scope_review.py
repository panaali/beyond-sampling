"""Independent moment counterexamples, saved scope and exact-prefix means."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,hashlib,math,time,mpmath as mp,torch
start=time.process_time();H=Path(__file__).parent;P=Path('/workspace/work/recover-small_h2_56/review/arc_cumulants_negative/recovery_review');INV=json.loads((P/'inventory.json').read_text());SRC=Path('/workspace/work/admin-recover56-cumulant-review/review/arc_cumulants_negative');manifest=json.loads((P/'source_manifest.json').read_text())
for r in manifest['files']:assert hashlib.sha256((SRC.parents[1]/r['path']).read_bytes()).hexdigest()==r['sha256']
assert len(INV['output_files'])==25 and len(INV['variant_rows'])==43;unique={}
for row in INV['variant_rows']:
 line=(SRC/row['path']).read_text().splitlines()[row['line']-1].split();assert line[0]=='T='+str(row['T']) and line[1]==row['model'] and line[2]=='truth' and float(line[3])==row['truth_printed'] and line[4]==row['variant'] and float(line[5])==row['estimate_printed'];key=(row['T'],row['model'],row['variant']);unique[key]=row
assert len(unique)==42
for g in INV['groups']:
 rr=[r for r in unique.values() if r['T']==g['T'] and r['variant']==g['variant']];assert len(rr)==g['models'];v=math.sqrt(sum((r['estimate_printed']-r['truth_printed'])**2 for r in rr)/len(rr));assert abs(v-g['rmse_from_rounded_printed_estimates'])<1e-15
A=json.loads((P/'analytic_controls.json').read_text())['moment_pair'];x=list(map(F,A['support']));wa=list(map(F,A['weights_a']));wb=list(map(F,A['weights_b']));assert all(sum((a-b)*v**j for a,b,v in zip(wa,wb,x))==0 for j in range(5));assert sum(a for a,v in zip(wa,x) if v>0)==F(5,16) and sum(b for b,v in zip(wb,x) if v>0)==F(11,16)
# Covariance eigenvalues follow directly from E[X+²]=1/2 and disjoint half-rays.
assert .5-1/math.pi>0 and abs((.5-1/(2*math.pi))**2-(1/(2*math.pi))**2-(.25-1/(2*math.pi)))<1e-16
proto=json.loads((P/'prefix_protocol.json').read_text());path=Path(proto['weight_path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==proto['weight_sha256'];sd=torch.load(path,weights_only=True,map_location='cpu');wi=[F(float(x)) for x in sd['rnn.weight_ih_l0'][:,0]];wh=[[F(float(x)) for x in r] for r in sd['rnn.weight_hh_l0']];D=json.loads((P/'prefix_trained.json').read_text());mp.mp.dps=80;m=lambda f:mp.mpf(f.numerator)/f.denominator;refs=[];gaussians=[];nneg=0
for i,row in enumerate(D['rows']):
 a=sum(v*max(w,F(0)) for v,w in zip(wh[i],wi));c=sum(v*max(-w,F(0)) for v,w in zip(wh[i],wi));b=wi[i];assert b and str(a)==row['a_plus_exact'] and str(c)==row['a_minus_exact'];aa,cc,bb=m(a),m(c),m(b);ref=(mp.sqrt(aa*aa+bb*bb)+aa+mp.sqrt(cc*cc+bb*bb)+cc)/(2*mp.sqrt(2*mp.pi));assert ref>0 and abs(ref-mp.mpf(row['angular_reference']))<mp.mpf('1e-45');refs.append(ref);nneg+=int(a<0)+int(c<0)
 mu=(aa+cc)/mp.sqrt(2*mp.pi);var=(aa*aa+cc*cc)/2+bb*bb-mu*mu;g=mp.sqrt(var/(2*mp.pi))*mp.exp(-mu*mu/(2*var))+mu*mp.erfc(-mu/mp.sqrt(2*var))/2;assert abs(g-mp.mpf(row['gaussian_mean']))<mp.mpf('1e-44');gaussians.append(g)
relative=mp.sqrt(sum((a-b)**2 for a,b in zip(refs,gaussians))/sum(v*v for v in refs));assert abs(relative-mp.mpf(D['gaussian_relative_rms_mean_error']))<mp.mpf('1e-45');expected=dict(comparison=64,negation=16,arithmetic=2*(16*16+16*15)+32+32+2*nneg,hypot=32);assert expected==D['kernel_counts'] and sum(expected.values())==1208
controls=json.loads((P/'prefix_controls.json').read_text());assert len(controls['rows'])==16 and controls['naive_zero_for_positive_reference']==2
for r in controls['rows']:
 a,b=mp.mpf(r['a']),mp.mpf(r['b']);ref=(mp.sqrt(a*a+b*b)+a)/mp.sqrt(2*mp.pi);assert abs(ref-mp.mpf(r['reference']))<mp.mpf('1e-40')*max(1,abs(ref))
out=dict(saved_output_files=25,unique_printed_rows=42,duplicate_rows=1,scalar_latent_relu_covariance_rank=2,four_moment_positive_probabilities=['5/16','11/16'],exact_prefix_units=16,all_input_coefficients_nonzero=True,all_exact_h2_means_positive=True,gaussian_relative_rms_mean_error=float(relative),kernel_named_units=1208,review_cpu=time.process_time()-start,scope='Corrected theorem and finite historical evidence scope, plus exact two-step mean specialization only. No network forward/random samples/final accuracy; downstream Gaussian closure remains unvalidated and finite old failures are preserved.')
(H/'cumulant_scope_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
