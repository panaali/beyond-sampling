"""Bounded exact-rational topology reference; floating mass explicitly separate."""
from fractions import Fraction
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from exact import Exact, dyadic_model

HERE = Path(__file__).resolve().parent
ROOT = Path('/workspace/clean-neurons')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def float_columns(rays):
    return np.array([[float(Fraction(x, sum(abs(v) for v in ray))) for x in ray] for ray in rays]).T


def serialize_simplex(item):
    rays, determinant = item
    return dict(integer_rays=[[str(x) for x in ray] for ray in rays],
                normalized_absolute_determinant=[str(determinant.numerator), str(determinant.denominator)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    H = args.hidden
    tag = 't4' if H == 2 else 'h4_t4'
    started = time.process_time()
    import torch
    checkpoint = ROOT/f'work/checkpoints/H{H}_T4_seed0.pth'
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    arrays = tuple(state[k].numpy().astype(float) for k in ('rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'linear.weight'))
    arrays = (arrays[0].ravel(), arrays[1], arrays[2])
    model = dyadic_model(arrays)
    exact = Exact(seconds=max(1., 110-(time.process_time()-started)))
    output = dict(complete=False, H=H, T=4, model_forward_inputs=0,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        classification='exact rational topology; floating Gaussian-mass integration remains numerical')
    def save():
        output['integer_work'] = dict(exact.counts)
        output['observed_bit_maxima'] = dict(exact.bits)
        output['cpu_seconds'] = time.process_time()-started
        output['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (HERE/f'results_{tag}.json').write_text(json.dumps(output, indent=2)+'\n')
    try:
        clock = time.process_time()
        cells, stage_counts = exact.gates(model)
        output['gate_enumeration_cpu'] = time.process_time()-clock
        output['gate_cells'] = len(cells)
        output['stage_counts'] = stage_counts
        gate_simplices, correct_simplices = [], []
        gate_det = Fraction(0)
        correct_det = Fraction(0)
        clock = time.process_time()
        for cone, A in cells:
            exact.certify(cone)
            for orthant in exact.orthants(cone):
                exact.certify(orthant)
                for simplex in exact.triangulate(orthant):
                    gate_simplices.append(simplex)
                    gate_det += simplex[1]
                    exact.observe((gate_det.numerator, gate_det.denominator), 'partition_accumulator', limit=False)
            for good in exact.outcomes(cone, A, model):
                exact.certify(good)
                for orthant in exact.orthants(good):
                    exact.certify(orthant)
                    for simplex in exact.triangulate(orthant):
                        correct_simplices.append(simplex)
                        correct_det += simplex[1]
                        exact.observe((correct_det.numerator, correct_det.denominator), 'correct_determinant_accumulator', limit=False)
        output['geometry_and_exact_audit_cpu'] = time.process_time()-clock
        output['gate_simplices'] = len(gate_simplices)
        output['correct_simplices'] = len(correct_simplices)
        output['exact_gate_determinant_sum'] = str(gate_det)
        output['correct_determinant_sum_float'] = float(correct_det)
        output['exact_correct_determinant_sum_hex'] = [hex(correct_det.numerator), hex(correct_det.denominator)]
        if gate_det != 16:
            raise ArithmeticError(f'exact gate partition determinant sum {gate_det}')
        save()
        # Independent floating DD gate enumeration, compared in its native order.
        dd_path = ROOT/'m16/approaches/exact_ladder/cones.py'
        dd = module('rational_reference_float_dd', dd_path)
        counts = [0]*4
        operations = dd.Ops()
        clock = time.process_time()
        floating = list(dd.iter_cells(arrays, operations, counts))
        comparable = counts == stage_counts and len(floating) == len(cells)
        ray_count_mismatches = 0
        incidence_mismatches = 0
        max_ray_difference = 0.
        max_map_difference = 0.
        for (cone, A), (R, Z, B, _) in zip(cells, floating):
            if not comparable or R.shape != (len(cone.rays), cone.dim):
                ray_count_mismatches += 1
                continue
            integers = [int(row[0])+(int(row[1]) << 64) for row in Z]
            reference = R/np.sum(abs(R), axis=1)[:, None]
            exact_rays = float_columns(cone.rays).T
            distances = np.max(abs(exact_rays[:, None, :]-reference[None, :, :]), axis=2)
            rows, columns = linear_sum_assignment(distances)
            if [integers[column] for column in columns] != list(cone.tight):
                incidence_mismatches += 1
            max_ray_difference = max(max_ray_difference, float(distances[rows, columns].max()))
            max_map_difference = max(max_map_difference, float(np.max(abs(B-np.array(A, dtype=float)))))
        output['dd_comparison'] = dict(cpu_seconds=time.process_time()-clock,
            source_sha256=hashlib.sha256(dd_path.read_bytes()).hexdigest(),
            stage_counts=counts, gate_counts_equal=comparable,
            ray_count_mismatches=ray_count_mismatches, incidence_mismatches=incidence_mismatches,
            exact_tight_sets_equal_after_numerical_ray_matching=comparable and not ray_count_mismatches and not incidence_mismatches,
            comparison_note='Matching only compares independent outputs; no floating decisions enter exact geometry',
            max_l1_ray_coordinate_difference=max_ray_difference, max_gate_map_difference=max_map_difference,
            stored_reference_correct_simplices=258 if H == 2 else 5040,
            matches_stored_simplex_count=len(correct_simplices)==(258 if H == 2 else 5040))
        # Equal nonzero readout rows must behave exactly like the zero readout.
        # Build the two outcome decompositions separately and compare integer cones.
        zero = (model[0], model[1], [[Fraction(0)]*H for _ in range(4)])
        equal = (model[0], model[1], [[Fraction(i+1) for i in range(H)] for _ in range(4)])
        control_simplices = []
        clock = time.process_time()
        for cone, A in cells:
            first = exact.outcomes(cone, A, zero)
            second = exact.outcomes(cone, A, equal)
            if [(p.rays, p.tight) for p in first] != [(p.rays, p.tight) for p in second]:
                raise ArithmeticError('nonzero duplicate-row tie handling differs')
            for good in first:
                for orthant in exact.orthants(good):
                    exact.certify(orthant)
                    control_simplices.extend(exact.triangulate(orthant))
        output['constant_tie_geometry_cpu'] = time.process_time()-clock
        output['constant_tie_simplices'] = len(control_simplices)
        # Persist integer rays before any floating integration.
        artifact = dict(checkpoint_sha256=output['checkpoint_sha256'],
            model=[[[str(x.numerator), str(x.denominator)] for x in model[0]],
                   [[[str(x.numerator), str(x.denominator)] for x in row] for row in model[1]],
                   [[[str(x.numerator), str(x.denominator)] for x in row] for row in model[2]]],
            gate_simplices=[serialize_simplex(s) for s in gate_simplices],
            correct_simplices=[serialize_simplex(s) for s in correct_simplices])
        (HERE/f'geometry_{tag}.json').write_text(json.dumps(artifact, separators=(',', ':'))+'\n')
        output['integer_artifact_bytes'] = (HERE/f'geometry_{tag}.json').stat().st_size
        volume_path = ROOT/'m16/approaches/small_h4_4/stable_volume.py'
        volume = module('rational_reference_positive_mass', volume_path)
        output['mass_source_sha256'] = hashlib.sha256(volume_path.read_bytes()).hexdigest()
        output['mass_rows'] = []
        for name, simplices, orders in (('accuracy', correct_simplices, (16, 20)),
                                        ('gate_partition', gate_simplices, (20,)),
                                        ('constant_class_zero', control_simplices, (20,))):
            rays = [float_columns(s[0]) for s in simplices]
            for order in orders:
                exact.check()
                clock = time.process_time()
                values = np.array([volume.mass(ray, order) for ray in rays])
                output['mass_rows'].append(dict(name=name, order=order,
                    probability=float(values[:, 0].sum()), quadrature_only_bound=float(values[:, 1].sum()),
                    nodes=len(rays)*order**3, cpu_seconds=time.process_time()-clock))
        output['complete'] = True
    except (ArithmeticError, ValueError, TimeoutError, AssertionError) as error:
        output['failure'] = repr(error)
    save()
    print(json.dumps(output))

if __name__ == '__main__':
    main()
