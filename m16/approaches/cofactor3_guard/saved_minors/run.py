"""Frozen64 stored cofactors, reusing original points without determinant replay."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from primitive import enclose
from controls import exact_reference


def rat(q):
    return [str(q.numerator), str(q.denominator)]


def main():
    assert not (HERE/'results.json').exists(), 'Preserve prior saved-minor results'
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    start = time.process_time()
    manifest = json.loads((HERE/'manifest.json').read_text())
    for path, pin in manifest['pins'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == pin
    assert (sys.float_info.radix, sys.float_info.mant_dig, sys.float_info.rounds) == (2, 53, 1)
    base = Path(next(p for p in manifest['pins'] if p.endswith('panel_nodes.npz')))
    rows = []
    with np.load(base) as archive:
        labels = sorted(int(n[:-3]) for n in archive.files if n.endswith('_Rt'))
        assert labels[:8] == manifest['labels']
        for label in manifest['labels']:
            rt = archive[str(label)+'_Rt']
            args = archive[str(label)+'_atan2_arguments']
            assert rt.shape == (128, 4, 4) and args.shape == (6, 2, 128)
            for node in manifest['nodes']:
                for pair in manifest['pair_indices']:
                    i, j = manifest['pair_order'][pair]
                    k, l = [q for q in range(4) if q != i and q != j]
                    ck = [q for q in range(4) if q != k]
                    cl = [q for q in range(4) if q != l]
                    sign = (-1)**(k+l)
                    a = [[float(rt[node, r, c]) for c in cl] for r in ck]
                    exact = exact_reference(a)
                    signed = sign*exact
                    got = enclose(a)
                    assert got['status'] != 'unknown_nonfinite'
                    lo, hi, point = got['lower'], got['upper'], got['point']
                    assert (lo == -math.inf or Fraction.from_float(lo) <= exact)
                    assert (hi == math.inf or exact <= Fraction.from_float(hi))
                    if sign < 0:
                        lo, hi, point = -hi, -lo, -point
                    assert (lo == -math.inf or Fraction.from_float(lo) <= signed)
                    assert (hi == math.inf or signed <= Fraction.from_float(hi))
                    original_y = float(args[pair, 0, node])
                    source = -original_y
                    assert math.isfinite(source) and math.isfinite(point)
                    error = Fraction.from_float(point)-signed
                    source_error = Fraction.from_float(source)-signed
                    comparison = 'tie' if abs(error) == abs(source_error) else (
                        'candidate_better' if abs(error) < abs(source_error) else 'source_better')
                    width = Fraction.from_float(hi)-Fraction.from_float(lo)
                    row = dict(label=label, node=node, pair_position=pair, pair=[i,j],
                               deleted_row=k, deleted_column=l, cofactor_sign=sign,
                               minor_hex=[[x.hex() for x in r] for r in a],
                               exact_unsigned=rat(exact), exact_signed=rat(signed),
                               saved_atan2_y=original_y.hex(), source_signed_point=source.hex(),
                               candidate_signed_point=point.hex(), interval=[lo.hex(), hi.hex()],
                               enclosed=True, contains_zero=lo <= 0 <= hi,
                               candidate_error_exact=rat(error), source_error_exact=rat(source_error),
                               absolute_error_comparison=comparison, interval_width_exact=rat(width),
                               relative_width=None if signed == 0 else rat(width/abs(signed)),
                               candidate_relative_error=None if signed == 0 else rat(abs(error/signed)),
                               source_relative_error=None if signed == 0 else rat(abs(source_error/signed)),
                               primitive_status=got['status'], primitive_counts=got['counts'],
                               signed_wrapper_negations=3 if sign < 0 else 0,
                               fallback_max_numerator_bits=got['fallback_max_numerator_bits'],
                               fallback_max_denominator_bits=got['fallback_max_denominator_bits'])
                    rows.append(row)
    assert len(rows) == manifest['count'] == 64
    summary = dict(cases=len(rows), cross_zero=sum(r['contains_zero'] for r in rows),
                   statuses={s: sum(r['primitive_status'] == s for r in rows)
                             for s in ('fast_enclosure', 'exact_fallback')},
                   point_comparisons={s: sum(r['absolute_error_comparison'] == s for r in rows)
                                      for s in ('candidate_better', 'source_better', 'tie')})
    for key in ('relative_width', 'candidate_relative_error', 'source_relative_error'):
        values=[(Fraction(int(r[key][0]),int(r[key][1])),idx) for idx,r in enumerate(rows) if r[key] is not None]
        value,idx=max(values)
        summary['max_'+key]=dict(value=rat(value), approximate=float(value), row=idx)
    result=dict(complete=True, manifest=manifest, rows=rows, summary=summary,
                cpu_seconds=time.process_time()-start, new_primitive_calls=64,
                new_native_determinant_mass_model_MC_calls=0,
                source_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in
                               [('run.py',Path(__file__)),('primitive.py',HERE.parent/'primitive.py'),
                                ('reference_controls.py',HERE.parent/'controls.py')]},
                scope='Fixed stored64 nonprincipal cofactors only. Native points reused from saved signed atan2 arguments; no source rerun or mass/speed/fullcorpus claim.')
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(summary=summary,cpu_seconds=result['cpu_seconds'])))


if __name__ == '__main__':
    main()
