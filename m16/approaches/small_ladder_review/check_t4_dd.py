"""Separate DD gate/outcome enumeration and pulling triangulation for T4."""
from pathlib import Path
import hashlib
import importlib.util
import json
import time
import numpy as np

HERE = Path(__file__).resolve().parent
APP = HERE.parent

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

dd = module('review_dd', APP/'exact_ladder/cones.py')
volume = module('review_volume', APP/'small_h4_4/stable_volume.py')

def split_orthants(R, Z, ops):
    pieces = [(R, Z)]
    for k in range(4):
        new = []
        for r, z in pieces:
            sides = dd.clip(r, z, np.eye(4)[k], 90+k, ops)
            new.extend(p for p in sides if p is not None and len(p[0]) >= 4)
        pieces = new
    return pieces

def simplices(R, Z, ops):
    out = []
    for r, z in split_orthants(R, Z, ops):
        r = r / np.sum(abs(r), axis=1)[:, None]
        for idx in dd.triangulate(r, z, ops):
            v = r[list(idx)].T
            if np.linalg.det(v) != 0:
                assert np.min(v, axis=1).dot(np.ones(4)) > -5  # finite bounded rays
                assert np.max(abs(v).sum(axis=0)-1) < 1e-12
                assert np.min(np.min(v, axis=1)*np.max(v, axis=1)) >= -1e-12
                out.append(v)
    return out

def main():
    import torch
    start = time.process_time()
    report = dict(method='Independent DD geometry; shared positive R4 volume',
                  source_sha256=hashlib.sha256((APP/'exact_ladder/cones.py').read_bytes()).hexdigest(), cases=[])
    for H in (2, 4):
        s = torch.load(f'/workspace/clean-neurons/work/checkpoints/H{H}_T4_seed0.pth', weights_only=True, map_location='cpu')
        model = tuple(s[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
        model = (model[0].ravel(),model[1],model[2])
        ops = dd.Ops(); counts = [0]*4; gate_det = 0.; rays = []; cells = 0
        begin = time.process_time()
        try:
            for cell in dd.iter_cells(model, ops, counts):
                cells += 1
                gate_det += sum(abs(np.linalg.det(v)) for v in simplices(cell[0],cell[1],ops))
                good, _ = dd.outcome_pieces(cell, model, ops, 60)
                for R,Z in good:
                    rays.extend(simplices(R,Z,ops))
                if time.process_time()-start > 40:
                    raise TimeoutError('review CPU budget')
            values = np.array([volume.mass(v, 16) for v in rays])
            row = dict(H=H,T=4,accuracy=float(values[:,0].sum()),quadrature_only_bound=float(values[:,1].sum()),
                       gate_determinant_sum=gate_det,gate_cells=cells,correct_simplices=len(rays),
                       dd_operation_counter=ops.n,cpu_seconds=time.process_time()-begin)
            assert abs(gate_det-16) < 1e-8, row
        except Exception as error:
            row = dict(H=H,T=4,failure=repr(error),gate_cells=cells,gate_determinant_sum=gate_det,cpu_seconds=time.process_time()-begin)
        report['cases'].append(row)
        report['cpu_seconds'] = time.process_time()-start
        (HERE/'t4_dd_review.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(row),flush=True)

if __name__ == '__main__':
    main()
