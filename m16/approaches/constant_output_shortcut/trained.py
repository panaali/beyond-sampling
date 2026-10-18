"""Only the two previously captured checkpoints; recognize/delegate without geometry."""
import time
START=time.process_time()
from pathlib import Path
import json,hashlib,resource
import numpy as np
import torch
import primitive as A
P=Path(__file__).resolve().parent

def digest(model):
 h=hashlib.sha256()
 for a in model:h.update(a.tobytes())
 return h.hexdigest()
def main():
 resource.setrlimit(resource.RLIMIT_CPU,(20,22));assert json.loads((P/'controls.json').read_text())['complete'];proto=json.loads((P/'protocol.json').read_text());priorproto=json.loads(Path(next(f for f in proto['pins'] if f.endswith('/capture_protocol.json'))).read_text());rows=[]
 for case in proto['training_cases']:
  weights=Path(case['weights']);assert hashlib.sha256(weights.read_bytes()).hexdigest()==priorproto['pins'][str(weights)]
  state=torch.load(weights,weights_only=True,map_location='cpu');model=(state['rnn.weight_ih_l0'].numpy()[:,0].astype(float),state['rnn.weight_hh_l0'].numpy().astype(float),state['linear.weight'].numpy().astype(float));before=digest(model);sentinel=object();calls=[]
  def fallback(m):calls.append(m is model);return sentinel
  out=A.dispatch(model,fallback);assert out['shortcut']['status']=='fallback' and calls==[True] and out['result'] is sentinel and digest(model)==before
  rows.append(dict(seed=case['seed'],checkpoint_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),model_sha256=before,recognizer=out['shortcut'],original_model_forwarded=True,original_result_object_returned=True,no_geometry_mass_replay=True))
 out=dict(complete=True,rows=rows,cpu=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),primitive_sha256=hashlib.sha256((P/'primitive.py').read_bytes()).hexdigest(),validation_only=True)
 assert not (P/'trained.json').exists();(P/'trained.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
