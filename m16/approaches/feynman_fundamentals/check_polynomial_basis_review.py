"""Frozen independent angular primitive substitution, one existing H2 model.

Alternatives: incorrect positive angular rule versus finite-basis error.
Use exact-degree Legendre product antiderivatives, unchanged shared branches
and radial Q64. No model expansion or basis selection from the result.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from numpy.polynomial.legendre import legmul, legint, legval
import torch

HERE = Path(__file__).resolve().parent
SOURCE = Path('/workspace/work/recover-small_h4_4/m16/approaches/anisotropic_harmonic/angular_polynomial_cycle')
sys.path.insert(0, str(SOURCE))
import driver
from poly_space import Space, PHI0, WIDTH


class AntiderivativeSpace(Space):
    def interval_gram(self, lo, hi):
        a = 2 * (np.asarray(lo) - PHI0) / WIDTH - 1
        b = 2 * (np.asarray(hi) - PHI0) / WIDTH - 1
        out = np.empty((len(a), self.nang, self.nang))
        for i in range(self.nang):
            ci = np.zeros(i + 1); ci[i] = 1
            for j in range(i, self.nang):
                cj = np.zeros(j + 1); cj[j] = 1
                primitive = legint(legmul(ci, cj))
                value = .5 * np.sqrt((2*i+1)*(2*j+1)) * (legval(b, primitive)-legval(a, primitive))
                out[:, i, j] = value
                out[:, j, i] = value
        return out


def main():
    start = time.process_time()
    reference = json.loads((SOURCE/'pilot.json').read_text())['rows'][0]
    modelpath = Path('/workspace/clean-neurons/work/checkpoints/H2_T3_seed0.pth')
    w = torch.load(modelpath, weights_only=True, map_location='cpu')
    model = (w['rnn.weight_ih_l0'].numpy().astype(float).ravel(),
             w['rnn.weight_hh_l0'].numpy().astype(float), w['linear.weight'].numpy().astype(float))
    check = AntiderivativeSpace()
    full = check.interval_gram(np.array([PHI0]), np.array([PHI0+WIDTH]))[0]
    left = check.interval_gram(np.array([PHI0]), np.array([PHI0+WIDTH/2]))[0]
    right = check.interval_gram(np.array([PHI0+WIDTH/2]), np.array([PHI0+WIDTH]))[0]
    identity_error = float(np.max(abs(full-np.eye(check.nang))))
    complement_error = float(np.max(abs(left+right-full)))
    driver.Space = AntiderivativeSpace
    value, stats = driver.estimate(model, quadrature_order=64, seconds=15)
    original = reference['polynomial']['estimate']
    result = dict(estimate=value, original=original, primitive_substitution_difference=value-original,
                  reference=reference['reference'], fourier=reference['fourier']['estimate'],
                  identity_error=identity_error, complement_error=complement_error,
                  cpu_seconds=time.process_time()-start, leaves=stats['leaves'],
                  checkpoint_sha256=hashlib.sha256(modelpath.read_bytes()).hexdigest(),
                  source_sha256={f:hashlib.sha256((SOURCE/f).read_bytes()).hexdigest() for f in ('poly_space.py','driver.py')},
                  scope='Independent angular antiderivative; shared branching and radial quadrature; no cost claim')
    assert identity_error < 2e-13 and complement_error < 2e-13
    assert abs(value-original) < 1e-10
    assert abs(value-reference['reference']) > abs(reference['fourier']['error'])
    (HERE/'polynomial_basis_review_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
