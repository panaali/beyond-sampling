"""Pure postprocessing of DRI v3 table rows; no model loading or evaluation.

C* here means largest tested requested budget passing the selected criterion.
Actual costs are retained. Missing coverage is unknown, not a measured failure.
No interpolation, extrapolation, monotonicity or simultaneous CI claim is made.
"""
import argparse
import json
import math
from pathlib import Path

DEFAULT_GROUPS = ('orig', 'seed', 's0.0025', 's0.005', 's0.01', 's0.02')
CLASSES = {'mechanistic', 'quadrature', 'sampling', 'unreviewed'}


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate(rows):
    if not isinstance(rows, list):
        raise ValueError('Expected a v3 table (JSON list), not legacy v2 data')
    keys, classes = set(), {}
    for r in rows:
        if not isinstance(r, dict):
            raise ValueError('Every table row must be an object')
        for field in ('name', 'cls', 'budget', 'group', 'cost', 'ratio', 'ci', 'tuned'):
            if field not in r:
                raise ValueError(f'Missing required field: {field}')
        if not isinstance(r['name'], str) or not r['name'] or not isinstance(r['group'], str) or not r['group']:
            raise ValueError('Method and group must be nonempty strings')
        if r['cls'] not in CLASSES or type(r['tuned']) is not bool:
            raise ValueError('Invalid reviewed class or tuned flag')
        if not _finite(r['budget']) or r['budget'] <= 0 or int(r['budget']) != r['budget']:
            raise ValueError('Requested budget must be a positive integer')
        if not _finite(r['cost']) or r['cost'] <= 0 or not _finite(r['ratio']):
            raise ValueError('Cost must be positive; cost and ratio must be finite')
        ci = r['ci']
        if ci is not None and (not isinstance(ci, (list, tuple)) or len(ci) != 2 or not all(_finite(x) for x in ci) or ci[0] > ci[1]):
            raise ValueError('CI must be null (unavailable) or two finite ordered numbers')
        key = r['name'], r['group'], r['budget']
        if key in keys:
            raise ValueError(f'Duplicate method/group/budget: {key}')
        keys.add(key)
        if r['name'] in classes and classes[r['name']] != r['cls']:
            raise ValueError(f'Inconsistent class for {r["name"]}')
        classes[r['name']] = r['cls']


def _passes(row, criterion):
    return not row['tuned'] and row['ratio'] <= 1 and (criterion == 'point' or (row['ci'] is not None and row['ci'][1] <= 1))


def _curve(by_budget, required_groups, criterion):
    tested = sorted(by_budget)
    eligible, passing, coverage = [], [], []
    for budget in tested:
        selected = by_budget[budget]
        missing = [g for g in required_groups if g not in selected]
        tuned = [g for g in required_groups if g in selected and selected[g]['tuned']]
        no_ci = [g for g in required_groups if g in selected and selected[g]['ci'] is None] if criterion == 'upper_ci' else []
        if missing or tuned or no_ci:
            coverage.append({'budget': budget, 'missing_groups': missing, 'tuned_groups': tuned, 'missing_ci_groups': no_ci})
            continue
        eligible.append(budget)
        if all(_passes(selected[g], criterion) for g in required_groups):
            passing.append(budget)
    best = max(passing) if passing else None
    best_rows = by_budget[best] if best is not None else {}
    return {
        'status': ('observed_match' if best is not None else
                   'no_match_on_eligible_grid' if eligible else 'insufficient_coverage'),
        'cstar_tested_budget': best,
        'actual_costs_at_cstar': {g: best_rows[g]['cost'] for g in required_groups} if best is not None else {},
        'passing_budgets': passing,
        'eligible_budgets': eligible,
        'coverage_gaps': coverage,
        'failed_lower_budgets': [b for b in eligible if best is not None and b < best and b not in passing],
        'passes_largest_eligible_budget': bool(eligible and eligible[-1] in passing),
    }


def summarize(rows, required_groups=DEFAULT_GROUPS):
    """Return descriptive per-class curves and a reviewed mechanistic headline.

Rows must already use the desired H,T, truth set and v3 accounting. This helper
cannot verify model-level completeness or a DRI class supplied in an input file.
"""
    validate(rows)
    required_groups = tuple(required_groups)
    if not required_groups or len(set(required_groups)) != len(required_groups) or not all(isinstance(g, str) and g for g in required_groups):
        raise ValueError('Required groups must be nonempty and unique')
    methods = []
    for name in sorted({r['name'] for r in rows}):
        selected = [r for r in rows if r['name'] == name]
        by_budget = {}
        for r in selected:
            by_budget.setdefault(r['budget'], {})[r['group']] = r
        groups = sorted({r['group'] for r in selected} | set(required_groups))
        methods.append({
            'name': name, 'cls': selected[0]['cls'],
            'all_required_groups': {c: _curve(by_budget, required_groups, c) for c in ('point', 'upper_ci')},
            'per_group': {g: {c: _curve(by_budget, (g,), c) for c in ('point', 'upper_ci')} for g in groups},
        })
    reviewed = [m for m in methods if m['cls'] == 'mechanistic']
    return {
        'definition': 'Largest tested requested budget passing at every required group; actual costs reported separately. No extrapolation.',
        'required_groups': list(required_groups),
        'reviewed_mechanistic_count': len(reviewed),
        'mechanistic_matches': {c: [m['name'] for m in reviewed if m['all_required_groups'][c]['cstar_tested_budget'] is not None]
                                for c in ('point', 'upper_ci')},
        'methods': methods,
        'limitations': [
            'Input table class/advice/coverage declarations are trusted; use a DRI-produced v3 table.',
            'Upper-CI criterion uses supplied per-row intervals, not simultaneous coverage across the search.',
            'Missing groups and absent budgets are unknown. No match on this grid is not impossibility.',
            'Group averages can hide individual model failures; model-level coverage is not in the table schema.',
        ],
    }


def render(result):
    lines = ['Crossover on the tested budget grid (actual FPE costs shown separately)',
             'Required groups: ' + ', '.join(result['required_groups'])]
    if result['reviewed_mechanistic_count'] == 0:
        lines.append('No DRI-reviewed mechanistic method in the input; headline remains unestablished.')
    for m in result['methods']:
        lines.append(f"\n{m['name']} [{m['cls']}]")
        for criterion, curve in m['all_required_groups'].items():
            value = curve['cstar_tested_budget']
            lines.append(f"  all groups / {criterion}: {value if value is not None else 'unknown or no tested match'} ({curve['status']})")
            if value is not None:
                lines.append('    actual cost by group: ' + json.dumps(curve['actual_costs_at_cstar'], sort_keys=True))
            if curve['failed_lower_budgets']:
                lines.append(f"    nonmonotone: failed lower budgets {curve['failed_lower_budgets']}")
            if curve['coverage_gaps']:
                lines.append(f"    incomplete/tuned/missing-CI budgets: {[g['budget'] for g in curve['coverage_gaps']]}")
        for group, curves in m['per_group'].items():
            lines.append(f"  {group}: point={curves['point']['cstar_tested_budget']}, upper_ci={curves['upper_ci']['cstar_tested_budget']}")
    lines.append('\nLimitations:')
    lines.extend('  ' + item for item in result['limitations'])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=Path, default=Path('results/leaderboard_v3_H16_T10.json'))
    parser.add_argument('--groups', default=','.join(DEFAULT_GROUPS))
    parser.add_argument('--out', type=Path, help='Optional JSON output; use this approach folder')
    args = parser.parse_args()
    if not args.table.is_file():
        parser.error(f'No v3 table at {args.table}; ask the DRI to run the v3 report. Legacy v2 rows cannot establish T19.')
    try:
        rows = json.loads(args.table.read_text())
        result = summarize(rows, tuple(g.strip() for g in args.groups.split(',')))
    except (ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    result['source_table'] = str(args.table)
    print(render(result))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
