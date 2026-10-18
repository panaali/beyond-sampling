"""Reconcile disjoint saved named-category subsets; does not rerun kernels."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
SPECS={
'q5':('q5_summary.json',['named_numeric_source_upper_total']),
'q4':('shift_review/contraction.json',['total','source_named_numeric_upper']),
'q3':('shift_review/parent_contraction.json',['total','parent_source_upper']),
'outer':('outer_review/contraction.json',['total','outer_source_upper']),
'application':('reduction_review/contraction.json',['total','application_source_upper']),
'blas':('reduction_review/blas_contraction.json',['total','child_upper','total']),
'driver':('driver_review/contraction.json',['total','source_upper'])}
def main():
 out=HERE/'composition.json';assert not out.exists()
 pins={};data={}
 for name,(f,keys) in SPECS.items():
  p=HERE/f;pins[f]=hashlib.sha256(p.read_bytes()).hexdigest();data[name]=(json.loads(p.read_text()),keys)
 p=HERE/'helper_review/contraction.json';pins[str(p.relative_to(HERE))]=hashlib.sha256(p.read_bytes()).hexdigest();helper=json.loads(p.read_text())
 modes={}
 for mode in ['exact','recipe108']:
  rows={}
  for name,(d,keys) in data.items():
   v=d['modes'][mode]
   for k in keys:v=v[k]
   assert isinstance(v,int) and v>=0;rows[name]=v
  h=helper['modes'][mode]['composed_named_subset']
  modes[mode]=dict(disjoint_other_components=rows,reflector_norm_helpers_conditional=h['conditional_total_upper'],reflector_norm_helpers_generic=h['generic_total_upper'],conditional_named_subset=sum(rows.values())+h['conditional_total_upper'],generic_named_subset=sum(rows.values())+h['generic_total_upper'])
 out.write_text(json.dumps(dict(modes=modes,pins=pins,scope='Disjoint previously reviewed named numeric source/native categories plus pending driver audit. Each component includes matching cold phase once. Conditional historical no-rescale binding and generic alternative separated. Larger cold eigensolver, integer/control/memory/casts/runtime and complete compiled lane correspondence unpriced. Neither a complete native cost nor an MSE multiplier.'),indent=2)+'\n')
 print(json.dumps(modes))
if __name__=='__main__':main()
