"""Independent torch recurrence and result-schema checks, no network requests."""
import json
import os
from pathlib import Path
import time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import torch
from alg_zoo.architectures import DistRNN

start = time.process_time()
folder = Path(__file__).parent
state = torch.load(folder / 'checkpoint.pth', weights_only=True, map_location='cpu')
model = DistRNN(16, 10).double().eval()
model.load_state_dict(state)
x = np.random.default_rng(82028).standard_normal((128, 10))
wi = state['rnn.weight_ih_l0'].numpy()[:, 0].astype(float)
wh = state['rnn.weight_hh_l0'].numpy().astype(float)
wo = state['linear.weight'].numpy().astype(float)
h = np.zeros((128, 16))
for t in range(10):
    h = np.maximum(h @ wh.T + x[:, t, None] * wi, 0)
with torch.no_grad():
    reference = model(torch.from_numpy(x)).numpy()
error = float(np.max(np.abs(reference-h@wo.T)))
assert error < 1e-10
for name in ('results.json', 'results_seed82027.json'):
    result = json.loads((folder/name).read_text())
    assert len(result['gate_checks']) == 5
    for row in result['gate_checks']:
        active = row['active']
        ev, v = np.linalg.eig(wh[np.ix_(active, active)])
        pairs = [j for j in range(len(ev)) if ev[j].imag > 1e-7]
        clock = max(pairs, key=lambda j: abs(ev[j]))
        high = sorted([j for j in pairs if np.angle(ev[j]) > np.angle(ev[clock])+1e-8], key=lambda j: -abs(ev[j]))[:2]
        raw = np.column_stack([part for j in high for part in (v[:, j].real, v[:, j].imag)])
        q, _ = np.linalg.qr(raw)
        lev = (q*q).sum(1)
        assert np.allclose(lev, row['orthogonal']['leverage_by_active_neuron'], atol=1e-10)
        assert np.isclose(lev.sum(), 4) and lev.min() >= 0 and lev.max() <= 1+1e-10
        assert row['ablation']['baseline_accuracy'] <= 1
out = {'torch_numpy_max_logit_error': error, 'qr_svd_leverages_agree': True,
       'validation_sequences': 256, 'gaussian_variates': 1280,
       'validation_recurrence_readout_flops': 256*(10*(2*16*16+3*16)+2*16*10),
       'eigendecompositions': 10, 'qr_decompositions': 10,
       'cpu_seconds_excluding_imports': time.process_time()-start}
(folder/'validation.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
