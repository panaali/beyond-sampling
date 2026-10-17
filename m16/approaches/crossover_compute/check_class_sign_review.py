"""Independent finite rank/sign accounting for T110; no neural code imported."""
from fractions import Fraction as F
from math import comb
import json
from pathlib import Path


def run():
    rows = []
    for t in range(3, 11):
        # Conditional on any fixed ordering, signs depend only on how many
        # of the iid symmetric continuous inputs are positive.
        rank_positive = [
            sum((F(comb(t, k), 2**t) for k in range(rank, t + 1)), F(0))
            for rank in range(1, t + 1)
        ]
        actual = [rank_positive[0], rank_positive[1],
                  sum(rank_positive[2:], F(0)) / (t - 2)]
        proposed = [1 - F(1, 2**t), 1 - F(t + 1, 2**t),
                    (F(t, 2) - 2 + F(t + 2, 2**t)) / (t - 2)]
        assert actual == proposed
        positive_total = F(0)
        all_total = F(0)
        for largest in range(t):
            for second in range(t):
                if largest == second:
                    continue
                p = actual[0 if largest == 0 else 1 if second == 0 else 2]
                positive_total += p / (t * (t - 1))
                all_total += (p + 1 - p) / (t * (t - 1))
        assert positive_total == F(1, 2) and all_total == 1
        rows.append(dict(T=t, conditional_positive=list(map(str, actual)),
                         positive_total=str(positive_total), all_total=str(all_total)))
    return dict(scope="Exact combinatorial chart masses; no clipping, neural accuracy or cost validation", rows=rows)


if __name__ == "__main__":
    result = run()
    Path(__file__).with_name("CLASS_SIGN_REVIEW.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Passed exact rank/sign identities and chart partitions for T=3..10.")
