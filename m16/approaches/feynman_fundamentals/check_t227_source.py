"""T55 independent byte/provenance and exact face-stratum source review.

No author modules, numerical walks, RNG, LPs, model loads, or tail contractions.
Run with the repository Python; output is immutable.
"""
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
OWNER = Path('/workspace/work/recover-small_h2_78')
BASE = OWNER / 'review/tie_tail_evidence'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    start = time.process_time()
    out = HERE / 't227_source_review.json'
    assert not out.exists()
    manifest = json.loads((BASE / 'source_manifest.json').read_text())
    assert len(manifest['files']) == 26
    checked = []
    for r in manifest['files']:
        p = OWNER / r['local_path']
        data = p.read_bytes()
        assert len(data) == r['bytes'] and sha(p) == r['sha256']
        git_data = subprocess.check_output(['git', '-C', str(OWNER), 'show', r['source_commit'] + ':' + r['source_path']])
        assert git_data == data
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        assert blob == r['git_blob']
        checked.append({'path': r['local_path'], 'sha256': sha(p), 'git_blob': blob, 'bytes': len(data)})
    # t = q*sqrt(2); movement in the pair is +/-q. All window
    # comparisons below divide out sqrt(2), so are rational and strict.
    q = F(3, 2)
    examples = []
    for kind, y, pair in [('12', [F(1), F(1), F(-4)], (0, 1)),
                          ('23', [F(4), F(-1), F(-1), F(-5)], (1, 2))]:
        a, b = pair
        u = y[a]
        others = sorted([v for j, v in enumerate(y) if j not in pair], reverse=True)
        window = (u - others[0]) / 3 if kind == '12' else min((others[0] - u) / 3, u - others[1])
        assert 0 < q < window == F(5, 3)
        x = y.copy(); x[a] += q; x[b] -= q
        k = sum(v > 0 for v in x)
        assert (u <= 0) != (k <= 1)
        examples.append({'kind': kind, 'y': list(map(str, y)), 'x': list(map(str, x)),
                         't_over_sqrt2': str(q), 'window_over_sqrt2': str(window),
                         'face_u_nonpositive': u <= 0, 'input_positive_count': k})
    result = {
        'task': 'T227 independent source/provenance review', 'complete': True, 'pass_all': True,
        'scope': 'Source claims and snapshot provenance only; not a saved numerical contraction or execution gate.',
        'owner_source_commit': '2f4c25c4', 'delegation_commit': '0b5a07d5',
        'manifest_sha256': sha(BASE / 'source_manifest.json'),
        'source_notes_sha256': sha(BASE / 'SOURCE_NOTES.md'), 'checker_sha256': sha(Path(__file__)),
        'files': checked, 'payload_bytes': sum(r['bytes'] for r in checked),
        'face_stratum_counterexamples': examples,
        'source_findings': {
            'primary': 'Finite distinct fixed-rule selected-tail coverage; no population coverage CI. Missing-kind filtering must be reported.',
            'n_check': 'Fresh Bernoulli thinning of already-selected rule A tail for each scheme/L; not a half-sized face-rule recomputation.',
            'null': 'Independent uniform K labels do not preserve the preregistered observed occupancy profile; singleton comparison is in-sample.',
            'within_label': 'Conditional within the already-selected global tail only; not the complete label-family second moment.',
            'flip_labels': 'Ordered first three events after rounded boundary keys and one minimizing gate; not an unordered complete set.',
            'face_vs_input': 'Whole-fibre face u<=0 stratum is distinct from actual-input K<=1; two strict-window rational examples checked.',
            'walks': 'Finite interpolated CDF tables, caps, eps shifts and numerical tie tolerance; scalar/vector Fcheck is same-source consistency.',
            'prefixes': 'Floating normalized HiGHS ranges; failed LP -> NaN can enter split branch. No saved exact feasible histories; sec is wall time.',
            'negative_scope': 'A verified low finite coverage can reject the frozen test, not all family decompositions or a mathematical tail-continuum theorem.'
        },
        'qualifications': [
            'Git-byte parity establishes snapshot identity, not original runtime/checkpoint identity.',
            'Original author ancestry was absent at the recorded inventory head; copied snapshots do not change that historical fact.',
            'No actual cap exhaustion, LP failure, trained numerical error, or full population result asserted.',
            'Random half/null assignments and complete all-face values are not reconstructed.',
            'T222 mass replication, T223 actual-event method, and T219 exact region recount remain separate.',
            'A separately frozen saved-only checker needs its own source/resource gate.'
        ],
        'author_calls': 0, 'model_calls': 0, 'rng_calls': 0, 'lp_calls': 0, 'mass_calls': 0,
        'review_cpu_seconds': time.process_time() - start
    }
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'complete': True, 'pass_all': True, 'files': len(checked), 'sha256': sha(out), 'cpu': result['review_cpu_seconds']}))

if __name__ == '__main__':
    main()
