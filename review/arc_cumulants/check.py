"""Bounded T42 audit of a pinned wrapper; no vendor imports or modifications.

Extracts the actual pure functions by AST because optional vendor dependencies
are absent locally. Does not replace their implementations with test doubles.
Only KH.get_d_max is supplied from its actual vendored function for cell_input.
"""
import ast
import hashlib
import itertools
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import time
from functools import cache
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch
from numpy.polynomial import Polynomial, hermite_e
from torch.utils.flop_counter import FlopCounterMode

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'm16/approaches/arc_cumulants/arc_kprop.py'
VENDOR = SOURCE.parent / 'vendor/mlp_kprop'


def extract(path, names, env):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes,
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), env)


def main():
    started = time.process_time()
    env = dict(np=np, os=os, ROOT=str(HERE), _CELL_CACHE={}, ID=0, RELU=1, HEAV=2)
    extract(SOURCE, ['rnn_layers', 'readout_uncond', 'readout_cell', 'forward_layers', 'cell_input',
                     'cell_base_cumulants'], env)
    d_env = dict(SIMPLE='simple', AUGMENT='augment', BASE='base')
    extract(VENDOR / 'kprop_harmonic.py', ['get_d_max'], d_env)
    env['KH'] = SimpleNamespace(SIMPLE='simple', get_d_max=d_env['get_d_max'])
    forward_layers = env['forward_layers']

    def exact(model, x):
        wi, wh, wo = model
        h = np.zeros((len(x), len(wi)))
        for t in range(x.shape[1]):
            h = np.maximum(h @ wh.T + x[:, t, None] * wi, 0)
        return h @ wo.T, h

    def compare(model, x):
        hsize = len(model[0]); T = x.shape[1]
        logits, hidden = exact(model, x)
        state = np.concatenate([np.zeros((len(x), hsize)), x], axis=1)
        carried = forward_layers(env['rnn_layers'](model), state)
        assert np.max(abs(carried[:, :hsize] - hidden)) < 1e-10
        assert np.array_equal(carried[:, hsize:], x)
        label = np.argsort(x, axis=1)[:, -2]
        target = logits.argmax(1) == label
        uncond = forward_layers(env['readout_uncond'](model), carried)[:, 0].astype(bool)
        cell = np.zeros(len(x), dtype=bool)
        for p2 in range(T):
            ids = np.flatnonzero(label == p2)
            cell[ids] = forward_layers(env['readout_cell'](model, p2), carried[ids])[:, 0].astype(bool)
        tied = (logits == logits.max(1, keepdims=True)).sum(1) > 1
        return dict(n=len(x), exact_accuracy=float(target.mean()), uncond_accuracy=float(uncond.mean()),
                    cell_accuracy=float(cell.mean()), uncond_mismatch=int((uncond != target).sum()),
                    cell_mismatch=int((cell != target).sum()), max_logit_tie_count=int(tied.sum()),
                    mismatch_without_tie=int(((uncond != target) & ~tied).sum()),
                    input_labels=label.tolist() if len(x) < 10 else None,
                    exact_bits=target.tolist() if len(x) < 10 else None,
                    uncond_bits=uncond.tolist() if len(x) < 10 else None,
                    cell_bits=cell.tolist() if len(x) < 10 else None)

    x = np.array(list(itertools.permutations([1., 2., 3.])))
    same = (np.ones(1), np.ones((1, 1)), np.ones((3, 1)))
    tied = compare(same, x)
    assert tied['uncond_mismatch'] == 4 and tied['cell_mismatch'] == 2
    rng = np.random.default_rng(42001)
    synthetic = compare((rng.normal(size=3), rng.normal(size=(3, 3)), rng.normal(size=(4, 3))),
                        rng.normal(size=(512, 4)))

    checkpoint = Path('/workspace/clean-neurons/m16/approaches/mechanism_checks/checkpoint.pth')
    s = torch.load(checkpoint, map_location='cpu', weights_only=True)
    base = (s['rnn.weight_ih_l0'].numpy()[:, 0].astype(float),
            s['rnn.weight_hh_l0'].numpy().astype(float), s['linear.weight'].numpy().astype(float))
    dev = []
    inputs = np.random.default_rng(42002).standard_normal((16384, 10))
    for sigma in (.0025, .02):
        noise = np.random.default_rng(1)
        model = tuple(w + sigma * np.sqrt((w*w).mean()) * noise.standard_normal(w.shape) for w in base)
        dev.append(dict(sigma=sigma, **compare(model, inputs)))

    # Actual public function, n below one chunk: no sampling or cache write occurs.
    try:
        env['cell_base_cumulants'](3, order=2, n=1, seed=42003)
        short_n = 'unexpected success'
    except Exception as exc:
        short_n = type(exc).__name__ + ': ' + str(exc)

    # Execute the wrapper's actual symmetrization statements on a small tensor.
    fn = next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.FunctionDef)
              and n.name == 'cell_base_cumulants')
    start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'perms' for t in n.targets))
    symbody = ast.Module(body=fn.body[start:start+3], type_ignores=[])
    matrix = np.zeros((6, 6)); matrix[2, 3] = matrix[3, 2] = 1
    se = dict(np=np, T=6, out={2: matrix})
    exec(compile(ast.fix_missing_locations(symbody), str(SOURCE), 'exec'), se)
    averaged = se['out'][2]; swap = [0, 1, 2, 4, 3, 5]
    symmetry_error = float(np.max(abs(averaged - averaged[np.ix_(swap, swap)])))
    assert symmetry_error == .25

    base_cumulants = {d: np.ones((3,) * d) for d in range(1, 5)}
    carried = env['cell_input'](1, 3, 0, 1, 2, base_cumulants)
    augment = dict(required_max_degree=d_env['get_d_max'](2, 'augment'),
                   actual_input_degrees=list(carried))
    assert augment['required_max_degree'] == 4 and augment['actual_input_degrees'] == [1, 2]

    # Vendor regularization and FLOP registration, without importing unavailable dependencies.
    wick = dict(torch=torch, math=math, cache=cache, Polynomial=Polynomial, hermite_e=hermite_e,
                logger=logging.getLogger('audit'))
    extract(VENDOR / 'wick.py', ['norm_pdf', 'norm_cdf', 'He', 'He_poly', 'eval_poly', '_relu_wick_poly',
                                'relu_wick_coef', 'heaviside_wick_coef'], wick)
    zero = torch.zeros(1, dtype=torch.float64)
    point_heav = float(wick['heaviside_wick_coef'](zero, zero, 0))
    point_relu = float(wick['relu_wick_coef'](zero, zero, 0))
    fw = dict(torch=torch, math=math, Any=Any, Callable=Callable, Iterable=Iterable,
              Optional=Optional, FlopCounterMode=FlopCounterMode, aten=torch.ops.aten)
    extract(VENDOR / 'flop_utils.py', ['_shape_numel', '_is_tensor_shape', '_out_numel', '_group_size',
            '_elementwise_1flop', '_add_like_flop', '_sum_like_flop', '_mean_flop', '_norm_flop',
            '_maybe', '_build_basic_mapping', 'ExtendedFlopCounterMode'], fw)
    fw['BASIC_FLOP_MAPPING'] = fw['_build_basic_mapping']()
    z = torch.linspace(-1, 1, 100, dtype=torch.float64)
    counted = {}
    for name, op in [('normal_cdf', torch.special.ndtr), ('erf', torch.erf), ('exp', torch.exp),
                     ('hermite4', lambda q: torch.special.hermite_polynomial_he(q, 4))]:
        with fw['ExtendedFlopCounterMode'](display=False) as counter:
            op(z)
        counted[name] = counter.get_total_flops()

    out = dict(source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               wrapper_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               identical_logits=tied, random_synthetic=synthetic, dev=dev,
               subchunk_failure=short_n, dihedral_symmetry_swap_error=symmetry_error,
               augmented_input_truncation=augment, zero_atom_heaviside=point_heav,
               zero_atom_relu_mean=point_relu, primitive_raw_flops=counted,
               model_inputs=2 * (2 * len(inputs)) + 2 * (len(x) + 512),
               cpu_seconds=time.process_time()-started,
               caveat='AST extracts exact source functions; no full cumulant propagation or source modifications.')
    (HERE / 'results.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
