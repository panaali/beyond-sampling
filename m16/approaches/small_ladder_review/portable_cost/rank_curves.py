"""Reprice saved quadrature curves with checked bounded-rank geometry.
No new model or numerical-integration evaluations. The GL16 geometry equivalence
check is explicit; other rules reuse identical ray-dependent integration work.
"""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
old=json.loads((HERE/'results.json').read_text());out=dict(models=[])
for row in old['models']:
 H=row['H'];new=json.loads((HERE/f'rank_checks_H{H}.json').read_text())['model']
 # Saved numerical checks predate this audit-only charge; no rerun required.
 rank_calls=new['geometry_cost']['events']['bounded_rank_calls']
 if 'qr_control_static' not in new['geometry_cost']['parts']:
  new['geometry_cost']['parts']['qr_control_static']=512*rank_calls
  new['geometry_cost']['known_arithmetic_upper_bound']+=512*rank_calls
 if 'determinant_control' not in new['geometry_cost']['parts']:
  new['geometry_cost']['parts']['determinant_control']=128*new['simplices']
  new['geometry_cost']['known_arithmetic_upper_bound']+=128*new['simplices']
 delta=new['geometry_cost']['known_arithmetic_upper_bound']-row['geometry_cost']['known_arithmetic_upper_bound']
 curves=[]
 for c in row['curves']:
  bound_charge_revision=50-c['cost']['parts']['bound']
  cost=c['cost']['known_arithmetic_upper_bound']+delta+bound_charge_revision+128*new['simplices'];fwd=c['conventional_forward_scalar_ops']
  samples=cost/fwd;var=c['accuracy']*(1-c['accuracy'])/samples
  curves.append(dict(order=c['order'],accuracy=c['accuracy'],integration_only_bound=c['integration_only_bound'],arithmetic_upper_bound=cost,bound_charge_revision=bound_charge_revision,conventional_forward_equivalents=samples,sampling_variance=var,bound_squared_over_sampling_variance=c['integration_only_bound']**2/var,quadrature_nodes=c['quadrature_nodes']))
  affine=4*(2*H*H+3*H-1);affine_samples=cost/affine;affine_var=c['accuracy']*(1-c['accuracy'])/affine_samples
  curves[-1].update(affine_only_forward_scalar_ops=affine,affine_only_forward_equivalents=affine_samples,affine_only_sampling_variance=affine_var,bound_squared_over_affine_only_sampling_variance=c['integration_only_bound']**2/affine_var)
 out['models'].append(dict(H=H,geometry_cost=new['geometry_cost'],production_black_box_kernels=0,curves=curves))
out['limitations']=['Conditional on unchanged floating geometry and numerical rank decisions; no total floating-error certificate.', 'Scalar arithmetic/comparison/bit operation cost1; transcendental cost20; allocation, copying and interpreter dispatch excluded.', 'Conventional forward equivalents exclude sampling generation, ranking and argmax: this comparison deliberately gives sampling free ancillary work.', 'Nominal checkpoints only; this cost audit does not establish the challenge criterion on independently perturbed model ensembles.', 'No CPU improvement inferred: validation timing includes SVD references and is separately saved.']
(HERE/'rank_curves.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps([dict(H=r['H'],GL16=r['curves'][3],GL20=r['curves'][4]) for r in out['models']],indent=2))
