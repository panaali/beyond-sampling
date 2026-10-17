"""Counted production evaluation of previously frozen noisy T4 checkpoints."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from estimator import calculate
HERE=Path(__file__).resolve().parent;ROOT=Path('/workspace/clean-neurons')
T47=HERE.parents[1]/'small_h2_34'
T51=Path('/workspace/work/recover-small_h4_4/m16/approaches/small_h4_4')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def model_hash(model):
 h=hashlib.sha256()
 for array in model:
  a=np.asarray(array,dtype='<f8');h.update(str(a.shape).encode('ascii'));h.update(a.tobytes(order='C'))
 return h.hexdigest()
def load(H):
 path=ROOT/f'work/checkpoints/H{H}_T4_seed0.pth'
 state=torch.load(path,weights_only=True,map_location='cpu')
 arrays=tuple(state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
 return (arrays[0].ravel(),*arrays[1:]),path

def main():
 start=time.process_time();cases=[];base4,checkpoint4=load(4)
 assert digest(checkpoint4)=='83ddefeaee737c73613cdc786b01b3bb79c0007bd6c1e42a454302c4801dd146'
 reference4=json.loads((T51/'cone_results.json').read_text());rng=np.random.default_rng(51001)
 for sigma,ref in zip((.0025,.02),reference4['models'][1:]):
  assert sigma==ref['sigma']
  model=tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in base4)
  q=next(row for row in ref['rules'] if row['order']==20)
  cases.append((f'H4_rms{sigma}',model,q['estimate'],q['integration_only_bound']))
 data2=json.loads((T47/'weights_seed0.json').read_text())['T4'];base2=tuple(np.array(data2[k]) for k in ('Wi','Wh','Wo'))
 restored2,checkpoint2=load(2)
 assert all(np.array_equal(a,b) for a,b in zip(base2,restored2))
 reference2=json.loads((T47/'stress_t4.json').read_text());rng=np.random.default_rng(47021)
 for sigma in (.02,.10):
  for i in range(4):
   name=f'rms{sigma}_copy{i}';ref=next(row for row in reference2['rows'] if row['name']==name)
   model=tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.normal(size=w.shape) for w in base2)
   cases.append(('H2_'+name,model,ref['reference'],ref['reference_bound']))
 provenance={str(p):digest(p) for p in (checkpoint4,checkpoint2,T51/'cone_reference.py',T51/'cone_results.json',T47/'stress_t4.py',T47/'stress_t4.json',T47/'weights_seed0.json')}
 output=dict(provenance=provenance,hash_encoding='SHA256 of concatenated str(shape).encode(ascii) and little-endian float64 C-order bytes, in Wi/Wh/Wo order',base_hashes=dict(H2=model_hash(base2),H4=model_hash(base4)),rows=[],random_model_inputs=0)
 output['models_generated_cpu_seconds']=time.process_time()-start
 for name,model,reference,refbound in cases:
  row=dict(name=name,model_sha256=model_hash(model),reference=reference,reference_bound=refbound);begin=time.process_time()
  try:
   p,stats=calculate(model);difference=p-reference
   assert not stats['cost']['black_box_kernels'] and not stats['cost']['validation_kernel_calls']
   consistent=abs(difference)<=stats['integration_only_bound']+refbound+1e-11
   sampling_variance=reference*(1-reference)/stats['affine_only_forward_equivalents']
   row.update(accuracy=p,difference=difference,within_combined_integration_bounds=consistent,stats=stats,affine_only_sampling_variance=sampling_variance,integration_bound_squared_over_sampling_variance=stats['integration_only_bound']**2/sampling_variance)
  except Exception as exc:
   row.update(error_type=type(exc).__name__,error=str(exc))
  row['evaluation_cpu_seconds']=time.process_time()-begin;output['rows'].append(row);output['total_cpu_seconds']=time.process_time()-start
  (HERE/'noise_checks.json').write_text(json.dumps(output,indent=2)+'\n')
  print(json.dumps({k:row[k] for k in ('name','accuracy','difference','within_combined_integration_bounds','error_type','error') if k in row}),flush=True)
  if output['total_cpu_seconds']>110:break
 print('TOTAL_CPU',output['total_cpu_seconds'],flush=True)
if __name__=='__main__':main()
