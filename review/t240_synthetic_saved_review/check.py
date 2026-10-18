"""Independent saved rational geometry/status audit; no author or CDF imports."""
from fractions import Fraction as F
from itertools import permutations, product
from pathlib import Path
import hashlib
import json
import math
import resource
import time
import traceback

HERE = Path(__file__).resolve().parent
START = time.process_time()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pincheck(pins):
    assert pins
    for p, h in pins.items():
        assert sha(p) == h, p


def fixed_cases():
    base = [[3, 4], [1, 2], [-1, 0]]
    rows = [('permutation' + str(n), 'box', [[base[j] for j in perm], 1])
            for n, perm in enumerate(permutations(range(3)))]
    rows += [(name, 'box', [box, gap]) for name, box, gap in [
        ('overlap_zero', [[0, 1]] * 3, 0), ('impossible', [[0, 1]] * 3, 1),
        ('touch_only', [[0, 1]] * 3, '1/2'),
        ('negative', [[-2, -1], [-4, -3], [-6, -5]], 1),
        ('adjacent_zero', [[2, 3], [1, 2], [0, 1]], 0),
        ('four_inputs', [[3, 4], [1, 2], [-1, 0], ['-1/2', '1/2']], '1/2')]]
    rows += [('margin' + str(n), 'margin', [target, v]) for n, (target, v) in enumerate([
        (0, [[0, 0], [0, 0], [0, 0]]), (1, [[0, 0], [0, 0], [1, 2]]),
        (2, [[0, 0], [0, 0], [0, 0]]), (1, [[1, 2], [0, 0], [0, 0]]),
        (1, [[-1, 1], [0, 0], [0, 2]]), (0, [[0, 0], [-2, -1], [-1, 1]])])]
    rows += [('mass' + str(n), 'mass', args) for n, args in enumerate([
        [['3/4', '4/5'], ['1/4', '3/10'], ['1/10', '1/8']],
        [[1, 1], [1, 1], [0, 0]], [[0, 0], [0, 0], [0, 0]],
        [['1/2', '3/4'], [0, 1], [0, 1]]])]
    rows += [(label, 'guard', args) for label, args in [
        ('negative_gap', [base, -1]), ('zero_width', [[[0, 0], [1, 2], [3, 4]], 0]),
        ('two_inputs', [[[0, 1], [2, 3]], 0]), ('float_endpoint', [[[0.0, 1], [2, 3], [4, 5]], 0]),
        ('target_range', [3, [[0, 0]] * 3]), ('self_margin', [0, [[1, 1], [0, 0]]]),
        ('negative_mass', [[-1, 0], [0, 0], [0, 0]]),
        ('inconsistent_mass', [[0, '1/2'], ['2/5', '1/2'], ['2/5', '1/2']]),
        ('reversed_mass', [[1, 0], [0, 0], [0, 0]])]]
    assert len(rows) == 31 and len({r[0] for r in rows}) == 31
    return rows


def gap_ok(x, i, j, g):
    return x[i] - x[j] >= g and all(x[j] - x[k] >= g for k in range(len(x)) if k not in (i, j))


def box_check(args, result, expected):
    box = [list(map(F, v)) for v in args[0]]
    gap = F(args[1]); d = len(box)
    assert result['box'] == [[str(v) for v in p] for p in box] and F(result['gap']) == gap
    vertices = list(product(*box)); inn = []; out = []; probes = 0
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            if all(gap_ok(x, i, j, gap) for x in vertices):
                inn.append([i, j])
            # Enumerate all critical second-coordinate endpoints, rather than
            # reuse the author's max/min formula or its supplied witness.
            critical = set(box[j]) | {box[i][1] - gap}
            critical |= {box[k][0] + gap for k in range(d) if k not in (i, j)}
            feasible = []
            for q in sorted(critical):
                probes += 1
                if not box[j][0] <= q <= box[j][1]:
                    continue
                if box[i][1] < q + gap:
                    continue
                if any(box[k][0] > q - gap for k in range(d) if k not in (i, j)):
                    continue
                feasible.append(q)
            if feasible:
                out.append(([i, j], [min(feasible), max(feasible)]))
    assert result['inner_pairs'] == expected['inner'] == inn
    assert [v['pair'] for v in result['outer_pairs']] == expected['outer'] == [v[0] for v in out]
    assert type(result['inner']) is bool and result['inner'] == bool(inn)
    assert type(result['outer']) is bool and result['outer'] == bool(out)
    for saved, (pair, ends) in zip(result['outer_pairs'], out, strict=True):
        assert list(map(F, saved['second_interval'])) == ends
        x = list(map(F, saved['witness']))
        assert len(x) == d and all(l <= v <= u for v, (l, u) in zip(x, box))
        assert gap_ok(x, *pair, gap)
        # Both interval endpoints admit a direct independent completion.
        for q in ends:
            y = [p[0] for p in box]; y[pair[1]] = q; y[pair[0]] = box[pair[0]][1]
            assert all(l <= v <= u for v, (l, u) in zip(y, box)) and gap_ok(y, *pair, gap)
    return dict(inner_pairs=inn, outer_pairs=[v[0] for v in out], vertex_pair_checks=len(vertices)*d*(d-1), critical_probes=probes, witnesses=len(out))


def margin_check(args, result, expected):
    target, v = args; v = [list(map(F, p)) for p in v]
    assert result['target'] == target and result['margins'] == [[str(x) for x in p] for p in v]
    winners = []
    for margins in product(*v):
        # Set the target logit to zero and reconstruct every rival logit.
        winner = max(range(len(v)), key=lambda o: (-margins[o], -o))
        winners.append(winner)
    status = 'correct' if all(w == target for w in winners) else 'wrong' if all(w != target for w in winners) else 'unknown'
    assert result['status'] == expected == status
    return dict(status=status, endpoint_winners=winners)


def mass_check(args, result, expected):
    p, c, w = [list(map(F, v)) for v in args]
    assert result['event'] == list(map(str, p)) and result['correct'] == list(map(str, c)) and result['wrong'] == list(map(str, w))
    # u=p-c-w, clipped only by the exact probability-domain bounds.
    u = [max(F(0), p[0] - (c[1] + w[1])), min(F(1), p[1] - (c[0] + w[0]))]
    cond = None if p[0] == 0 else [c[0] / p[1], (p[1] - w[0]) / p[1]]
    assert list(map(F, result['unresolved'])) == list(map(F, expected[0])) == u
    assert (None if result['conditional_accuracy'] is None else list(map(F, result['conditional_accuracy']))) == cond
    assert (None if expected[1] is None else list(map(F, expected[1]))) == cond
    # Endpoint realizations are supplemental checks, not an exhaustive event law.
    checked = 0
    for pv, cv, wv in product(p, c, w):
        if cv + wv > pv:
            continue
        checked += 1
        assert u[0] <= pv - cv - wv <= u[1]
        if cond is not None:
            assert cond[0] <= cv / pv and (pv - wv) / pv <= cond[1]
    assert checked
    return dict(unresolved=list(map(str, u)), conditional=None if cond is None else list(map(str, cond)), consistent_endpoint_triples=checked)


def guard_check(label, args, row):
    assert row['expected'] == 'ValueError' and 'result' not in row and 'unexpected_result' not in row
    assert isinstance(row['error'], str) and row['error']
    if label == 'negative_gap': assert args[1] < 0
    elif label == 'zero_width': assert any(p[0] == p[1] for p in args[0])
    elif label == 'two_inputs': assert len(args[0]) < 3
    elif label == 'float_endpoint': assert type(args[0][0][0]) is float
    elif label == 'target_range': assert args[0] >= len(args[1])
    elif label == 'self_margin': assert args[1][args[0]] != [0, 0]
    elif label == 'negative_mass': assert args[0][0] < 0
    elif label == 'inconsistent_mass': assert F(args[1][0]) + F(args[2][0]) > F(args[0][1])
    elif label == 'reversed_mass': assert args[0][0] > args[0][1]
    else: raise AssertionError(label)
    return dict(expected_exception='ValueError', saved_message=row['error'])


def run(out):
    scope = read(HERE/'scope.json'); gate = read(HERE/'gate.json')
    assert gate['ready'] is True and gate['review_publication']
    review = read(gate['source_review'])
    assert review['complete'] is True and review['pass_all'] is True and review['execution_authorized'] is True
    need = {str(HERE/n) for n in ['check.py', 'scope.json']}
    assert need <= set(review['pins'])
    assert need | {gate['source_review'], str(HERE/'completed_inputs.json')} <= set(gate['pins'])
    for pins in (gate['pins'], review['pins'], scope['pins']): pincheck(pins)
    completed = read(HERE/'completed_inputs.json'); pincheck(completed['pins'])
    assert completed['raw_commit'] and completed['publication']
    base = Path(scope['author'])
    required = {str(base/n) for n in ['primitive.py', 'run.py', 'PROTOCOL.md', 'protocol.json', 'launch.sh', 'gate.json', 'output/results.json', 'output/progress.jsonl', 'run1.log']}
    assert required <= set(completed['pins'])
    protocol = read(base/'protocol.json'); author_gate = read(base/'gate.json')
    author_review = read(protocol['source_review'])
    assert protocol['controls'] == 31 and protocol['expected_events'] == 64
    assert author_gate['ready'] is True and author_review['complete'] and author_review['pass_all'] and author_review['execution_authorized']
    assert str(Path(protocol['source_review'])) in completed['pins']
    for pins in (protocol['pins'],author_gate['pins'],author_review['pins']): pincheck(pins)
    r = read(base/'output/results.json')
    assert r['complete'] is True and r['pass_all'] is True and r['pins'] == author_gate['pins']
    assert r['author_model_calls'] == r['probability_calls'] == 0
    assert math.isfinite(r['cpu_seconds']) and r['cpu_seconds'] >= 0 and r['maxrss_kib'] > 0
    assert all(0 < v <= 20 for v in r['limits']['cpu']) and all(0 < v <= 2*1024**3 for v in r['limits']['as'])
    assert all(0 < v <= 20 for v in resource.getrlimit(resource.RLIMIT_CPU))
    assert all(0 < v <= 2*1024**3 for v in resource.getrlimit(resource.RLIMIT_AS))
    cases = fixed_cases(); assert len(r['rows']) == len(cases)
    events = [dict(kind='started')]; out['rows'] = []
    counts = dict(box=0, margin=0, mass=0, guard=0)
    for (label, kind, args), row in zip(cases, r['rows'], strict=True):
        assert time.process_time()-START < 19
        assert row['label'] == label and row['args'] == args and row['status'] == 'passed'
        assert json.dumps(row['args'], sort_keys=True) == json.dumps(args, sort_keys=True), 'Preserve float-versus-integer guard input types'
        if kind == 'guard': checked = guard_check(label, args, row)
        else: checked = {'box':box_check,'margin':margin_check,'mass':mass_check}[kind](args,row['result'],row['expected'])
        out['rows'].append(dict(label=label,kind=kind,checked=checked)); counts[kind] += 1
        events += [dict(kind='started_case',label=label),dict(kind='completed_case',label=label)]
    events.append(dict(kind='completed'))
    assert [json.loads(s) for s in (base/'output/progress.jsonl').read_text().splitlines()] == events
    assert len(events) == 64 and counts == dict(box=12,margin=6,mass=4,guard=9)
    out.update(complete=True,pass_all=True,counts=counts,events=64,author_cpu=r['cpu_seconds'],author_maxrss_kib=r['maxrss_kib'],
               input_pins=completed['pins'],author_calls=0,limitation='Synthetic rational ownership/bookkeeping only, not actual Gaussian event mass or trained correctness. Supplied interval endpoints and bookkeeping bounds remain premises.')


if __name__ == '__main__':
    path = HERE/'results.json'; assert not path.exists(), 'Preserve any prior attempt'
    result = dict(complete=False,pass_all=False)
    path.write_text(json.dumps(result)+'\n')
    try: run(result)
    except BaseException as exc:
        result.update(complete=False,pass_all=False,error=repr(exc),traceback=traceback.format_exc()); raise
    finally:
        result['review_cpu']=time.process_time()-START
        result['review_maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temp=HERE/'results.tmp';temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(path)
