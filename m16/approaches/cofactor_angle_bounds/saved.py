"""Frozen original64 numerator intervals, fixed stored denominators; no determinant replay."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,json,time
import numpy as np
import mpmath as mp
from bound import enclose,pack,serial
HERE=Path(__file__).resolve().parent

def frac(x):return F(int(x[0]),int(x[1]))
def m(x):return mp.mpf(x.numerator)/x.denominator

def main():
 out=HERE/'saved_results.json';progress=HERE/'saved_progress.jsonl';assert not out.exists() and not progress.exists()
 controls=json.loads((HERE/'controls.json').read_text());assert controls['cases']==21 and controls['failures']==[]
 protocol=json.loads((HERE/'protocol.json').read_text());assert hashlib.sha256((HERE/'bound.py').read_bytes()).hexdigest()==protocol['bound_sha256']
 saved_protocol=json.loads((HERE/'saved_protocol.json').read_text())
 assert hashlib.sha256((HERE/'controls.py').read_bytes()).hexdigest()==protocol['controls_sha256']
 assert hashlib.sha256((HERE/'controls.json').read_bytes()).hexdigest()==saved_protocol['completed_controls_sha256']
 assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==saved_protocol['saved_driver_sha256']
 frozen=json.loads((HERE/'manifest.json').read_text())
 for name,d in frozen['files'].items():assert hashlib.sha256((HERE/'frozen'/name).read_bytes()).hexdigest()==d['sha256']
 old=json.loads((HERE/'frozen/saved_minors_results.json').read_text());man=old['manifest'];assert old['complete'] and len(old['rows'])==64
 for path,pin in man['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==pin
 npz=Path(next(p for p in man['pins'] if p.endswith('panel_nodes.npz')))
 expected=[(label,node,pair) for label in man['labels'] for node in man['nodes'] for pair in man['pair_indices']]
 assert [(r['label'],r['node'],r['pair_position']) for r in old['rows']]==expected
 mp.mp.dps=120;rows=[];counts=Counter();statuses=Counter();bounds=[];peak=[0,0];fail=[];refs=0;start=time.process_time()
 with np.load(npz) as archive:
  # Materialize each used array once; avoid per-row ZIP decompression.
  args={label:archive[str(label)+'_atan2_arguments'] for label in man['labels']}
  for oldrow in old['rows']:
   label,node,pair=oldrow['label'],oldrow['node'],oldrow['pair_position']
   with progress.open('a') as f:f.write(json.dumps(dict(stage='attempt',label=label,node=node,pair=pair))+'\n')
   a=args[label];assert a.shape==(6,2,128)
   xfloat=float(a[pair,1,node]);source_y=float(a[pair,0,node]);assert source_y.hex()==oldrow['saved_atan2_y']
   x=F.from_float(xfloat) if np.isfinite(xfloat) else None
   clo,chi=[F.from_float(float.fromhex(v)) for v in oldrow['interval']]
   lo,hi=-chi,-clo;p=-F.from_float(float.fromhex(oldrow['candidate_signed_point']));truth=-frac(oldrow['exact_signed']);src=F.from_float(source_y)
   assert lo<=truth<=hi # reused certified exact numerator, no determinant recomputation
   got=enclose(x,lo,hi,p);counts.update(got['counts']);statuses[got['status']]+=1
   peak=[max(peak[i],got['peak_bits'][i]) for i in range(2)];diagnostic=None
   if got['status'].startswith('bounded'):
    bounds.append(got['bound']);exactangle=mp.atan2(m(truth),m(x));candidate=mp.atan2(m(p),m(x));original=mp.atan2(m(src),m(x));refs+=3
    error=abs(candidate-exactangle);sourceerror=abs(original-exactangle);inside=error<=m(got['bound'])+mp.mpf('1e-110')
    if not inside:fail.append([label,node,pair])
    comparison='tie' if error==sourceerror else ('candidate_better' if error<sourceerror else 'source_better')
    diagnostic=dict(candidate_angle_error=mp.nstr(error,110),source_angle_error=mp.nstr(sourceerror,110),inside_bound=bool(inside),comparison=comparison)
   row=dict(label=label,node=node,pair=pair,x_hex=xfloat.hex(),signed_y_interval=[serial(lo),serial(hi)],candidate_y=serial(p),exact_y=serial(truth),bound_result=pack(got),angle_reference=diagnostic,additional_signed_wrapper_negations=3,input_dyadic_conversions=4,inherited_primitive_counts=oldrow['primitive_counts'],inherited_cofactor_sign_negations=oldrow['signed_wrapper_negations'],inherited_primitive_status=oldrow['primitive_status'])
   rows.append(row)
   with progress.open('a') as f:f.write(json.dumps(row)+'\n')
 result=dict(cases=len(rows),failures=fail,statuses=dict(statuses),max_bound=serial(max(bounds)) if bounds else None,rows=rows,total_bound_categories=dict(counts),peak_bits=peak,additional_signed_wrapper_negations=3*len(rows),input_dyadic_conversions=4*len(rows),reference_angle_calls=refs,cpu_seconds=time.process_time()-start,new_determinant_model_mass_geometry_MC_calls=0,scope='Numerator-only exact rational bound at stored denominator. High-precision angle references diagnostic, not libm or fullmass certificate. Original64 cases only; every unknown retained; rational operation/bit costs not converted into native units.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['cases','failures','statuses','max_bound','cpu_seconds']}))
if __name__=='__main__':main()
