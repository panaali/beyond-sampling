"""Scientific failure-mode checks for the crossover summary; no model calls."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('crossover_report', Path(__file__).with_name('report.py'))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def row(group='orig', budget=100, ratio=0.5, upper=0.8, **kw):
    return dict(name='candidate', cls='mechanistic', budget=budget, group=group,
                cost=budget * 1.2, ratio=ratio, ci=[ratio - 0.2, upper], tuned=False) | kw


class CrossoverChecks(unittest.TestCase):
    def curve(self, rows, groups=('orig',), criterion='point'):
        return report.summarize(rows, groups)['methods'][0]['all_required_groups'][criterion]

    def test_missing_group_is_unknown(self):
        c = self.curve([row()], ('orig', 'seed'))
        self.assertIsNone(c['cstar_tested_budget'])
        self.assertEqual(c['status'], 'insufficient_coverage')

    def test_common_budget_required(self):
        c = self.curve([row(), row('seed', 400)], ('orig', 'seed'))
        self.assertIsNone(c['cstar_tested_budget'])

    def test_nonmonotone_and_actual_cost_preserved(self):
        c = self.curve([row(budget=100, ratio=1.2, upper=1.5), row(budget=400)])
        self.assertEqual(c['cstar_tested_budget'], 400)
        self.assertEqual(c['failed_lower_budgets'], [100])
        self.assertEqual(c['actual_costs_at_cstar'], {'orig': 480})

    def test_point_and_interval_separate(self):
        rows = [row(upper=1.2)]
        self.assertEqual(self.curve(rows)['cstar_tested_budget'], 100)
        self.assertIsNone(self.curve(rows, criterion='upper_ci')['cstar_tested_budget'])

    def test_advice_exclusion(self):
        c = self.curve([row(), row('seed', tuned=True)], ('orig', 'seed'))
        self.assertEqual(c['status'], 'insufficient_coverage')

    def test_unreviewed_never_mechanistic_headline(self):
        r = report.summarize([row(cls='unreviewed')], ('orig',))
        self.assertEqual(r['reviewed_mechanistic_count'], 0)
        self.assertEqual(r['mechanistic_matches']['point'], [])
        self.assertEqual(r['methods'][0]['all_required_groups']['point']['cstar_tested_budget'], 100)

    def test_bad_data_and_legacy_rejected(self):
        for rows in ({'v2|method|orig|100': {}}, [row(), row()], [row(ratio=float('nan'))],
                     [row(cost=0)], [row(ci=[1, 0])], [row(tuned='false')],
                     [row(), row('seed', cls='sampling')]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                report.summarize(rows)

    def test_no_extrapolation_and_no_eligible_win(self):
        c = self.curve([row(), row(budget=400, ratio=2, upper=3)])
        self.assertEqual(c['cstar_tested_budget'], 100)
        self.assertFalse(c['passes_largest_eligible_budget'])
        c = self.curve([row(ratio=2, upper=3)])
        self.assertEqual(c['status'], 'no_match_on_eligible_grid')

    def test_negative_truth_corrected_mse_allowed(self):
        c = self.curve([row(ratio=-0.01, upper=0.1)])
        self.assertEqual(c['cstar_tested_budget'], 100)

    def test_missing_interval_is_unknown_not_failure(self):
        c = self.curve([row(ci=None)])
        self.assertEqual(c['cstar_tested_budget'], 100)
        c = self.curve([row(ci=None)], criterion='upper_ci')
        self.assertEqual(c['status'], 'insufficient_coverage')
        self.assertEqual(c['coverage_gaps'][0]['missing_ci_groups'], ['orig'])

    def test_empty_data_and_boundary(self):
        self.assertEqual(report.summarize([])['methods'], [])
        c = self.curve([row(ratio=1, upper=1)], criterion='upper_ci')
        self.assertEqual(c['cstar_tested_budget'], 100)


if __name__ == '__main__':
    unittest.main()
