"""Fixed saved-only T198 reconciliation. No author imports or numerical replays."""
import hashlib
import json
import math
from pathlib import Path
from fractions import Fraction as F
import re
import time
import traceback

ROOT = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    return json.loads(p.read_text())


def exact(x):
    assert isinstance(x, (int, float)) and not isinstance(x, bool)
    assert math.isfinite(x)
    return F(x)


def enc(x):
    return {'numerator': str(x.numerator), 'denominator': str(x.denominator), 'float': float(x)}


def main():
    started = time.process_time()
    outputs = [ROOT / n for n in ('saved_results.json', 'saved_progress.jsonl', 'saved_failure.json')]
    assert not any(p.exists() for p in outputs), 'preserve any prior attempt'
    gate = load(ROOT / 'execution_gate.json')
    assert gate['status'] == 'completed_source_review'
    assert gate['review_publication'] and gate['review_artifact']
    for name, digest in gate['pins'].items():
        p = Path(name)
        if not p.is_absolute():
            p = ROOT / p
        assert sha(p) == digest, name
    with outputs[1].open('x') as out:
        out.write(json.dumps({'event': 'started', 'gate_sha256': sha(ROOT / 'execution_gate.json')}) + '\n')
        out.flush()
        try:
            manifest = load(ROOT / 'manifest.json')
            assert len(manifest['records']) == 73
            paths = {}
            for row in manifest['records']:
                p = ROOT / 'frozen' / row['original']
                assert sha(p) == row['sha256'] and p.stat().st_size == row['bytes']
                assert row['original'] not in paths
                paths[row['original']] = p
            prefix = 'm16/approaches/split_attribution/'
            raw = sorted(p for p in paths if p.startswith(prefix + 'prune/'))
            assert len(raw) == 51 and raw == sorted(manifest['prune_paths'])
            truth = load(paths[prefix + 'ref/exact_H8_T3.json'])
            keys = set()
            rows = []
            groups = {}
            for name in raw:
                r = load(paths[name])
                assert r['H'] == 8 and r['T'] == 3
                assert type(r['seed']) is int and r['seed'] in range(5)
                assert r['test'] in ('cap', 'mass')
                key = (r['seed'], r['test'], r['tau'])
                assert key not in keys
                keys.add(key)
                for field in ('ops', 'flops', 'retained_cells', 'resolved', 'pruned', 'wrong_centroid'):
                    assert type(r[field]) is int and r[field] >= 0
                assert r['flops'] == 2 * r['ops']
                assert r['wrong_centroid'] <= r['pruned']
                assert r['sec'] >= 0 and math.isfinite(r['sec'])
                assert sum(v[0] for v in r['by_step'].values()) == r['pruned']
                for step, value in r['by_step'].items():
                    assert step in ('1', '2', '3') and len(value) == 2
                    assert type(value[0]) is int and value[0] >= 0
                    exact(value[1])
                p = exact(truth[str(r['seed'])])
                assert 0 < p < 1
                a, b, flops = exact(r['acc']), exact(r['bound']), exact(r['flops'])
                assert 0 <= a <= 1 and b >= 0 and flops > 0
                n = flops / 511
                variance = p * (1-p) / n
                ra, rb = (a-p)**2 / variance, b*b / variance
                by_sum = sum((exact(v[1]) for v in r['by_step'].values()), F(0))
                pad = exact(1e-15) * r['pruned'] if r['test'] == 'mass' else F(0)
                ref = load(paths[prefix + f'ref/H8_T3_s{r["seed"]}.json'])
                assert ref['cells'] > 0 and ref['ops'] > 0
                historical_n_residual = exact(r['N']) - flops/500
                assert abs(historical_n_residual) <= exact(math.ulp(r['N']))
                entry = dict(name=name, sha256=sha(paths[name]), key=list(key),
                             estimate_error=enc(a-p), bound=enc(b), canonical_continuous_draws=enc(n),
                             historical_N_residual=enc(historical_n_residual),
                             canonical_variance=enc(variance), ratio_actual=enc(ra), ratio_bound=enc(rb),
                             padded_bound_minus_step_sums=enc(b-by_sum-pad),
                             retained_cells=r['retained_cells'], reference_cells=ref['cells'],
                             flops=r['flops'], pruned=r['pruned'], author_wall_seconds=r['sec'])
                rows.append(entry)
                if r['tau']:
                    groups.setdefault((r['test'], r['tau']), []).append((r['seed'], ra, rb, r['retained_cells'], ref['cells'], r['flops']))
                out.write(json.dumps({'event': 'row_checked', 'name': name}) + '\n')
                out.flush()
            taus = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
            expected = {(s,t,q) for s in range(5) for t in ('cap','mass') for q in taus}
            expected.add((0,'cap',0))
            assert keys == expected
            pooled = []
            for key, values in sorted(groups.items()):
                assert len(values) == 5 and {v[0] for v in values} == set(range(5))
                pooled.append(dict(test=key[0], tau=key[1], seeds=[v[0] for v in values],
                                   max_ratio_actual=enc(max(v[1] for v in values)),
                                   max_ratio_bound=enc(max(v[2] for v in values)),
                                   mean_cells=enc(F(sum(v[3] for v in values),5)),
                                   mean_reference_cells=enc(F(sum(v[4] for v in values),5)),
                                   mean_flops=enc(F(sum(v[5] for v in values),5))))
            assert len(pooled) == 10
            text = paths[prefix+'capcheck.txt'].read_text().splitlines()
            match = re.fullmatch(r'checks (\d+) violations (\d+)', text[-1])
            assert match and tuple(map(int,match.groups())) == (834,0)
            assert len(text)-1 == 135
            result = dict(status='pass', manifest_sha256=sha(ROOT/'manifest.json'),
                          checker_sha256=sha(Path(__file__)), gate_sha256=sha(ROOT/'execution_gate.json'),
                          rows=rows, groups=pooled, rows_checked=len(rows), groups_checked=len(pooled),
                          printed_control_summary={'checks':834,'violations':0,'retained_lines':135,
                              'not_independently_replayed':True,'all_rays_and_values_absent':True},
                          T4_T5_replay_files=0, rendered_report_present=manifest['rendered_prune_report_present'],
                          bound_step_residual_scope='diagnostic only: no individual summands or outward accumulation certificate',
                          comparison_scope='exact stored-float point references and continuous T196 draws; no CI or complete-cost claim',
                          author_wall_seconds=sum(r['author_wall_seconds'] for r in rows),
                          review_process_cpu=time.process_time()-started,
                          author_cap_geometry_mass_model_RNG_calls=0)
            assert result['rendered_report_present'] is False
            with outputs[0].open('x') as f:
                json.dump(result,f,indent=2);f.write('\n')
            out.write(json.dumps({'event':'completed','result_sha256':sha(outputs[0])})+'\n')
            out.flush()
        except BaseException:
            with outputs[2].open('x') as f:
                json.dump({'traceback':traceback.format_exc(),'review_process_cpu':time.process_time()-started},f,indent=2)
            out.write(json.dumps({'event':'failed','failure_sha256':sha(outputs[2])})+'\n');out.flush()
            raise


if __name__ == '__main__':
    main()
