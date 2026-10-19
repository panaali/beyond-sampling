"""F per architecture group, estimator and budget (pre-registered formula) -> results.md."""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "results", "*.json")))]
groups = sorted({r["name"].split("_")[1] for r in R})
keys = sorted({k for r in R for k in r["runs"]}, key=lambda k: (int(k.split("@")[1]), k))
rng = np.random.default_rng(0)


def F(rs, k):
    """(point F, bootstrap 95% CI) over models: mean(mse - truth var) * C_eff / mean(sigma2)."""
    mse = np.array([np.mean([(x["est"] - r["truth"]) ** 2 for x in r["runs"][k]]) - r["sigma2"] / r["n_truth"] for r in rs])
    s2 = np.array([r["sigma2"] for r in rs])
    C = np.array([np.mean([x["cost"] for x in r["runs"][k]]) for r in rs])
    Ce = max(float(np.mean(C)), 1e3)
    f = lambda i: mse[i].mean() * Ce / s2[i].mean()
    bs = [f(rng.integers(0, len(rs), len(rs))) for _ in range(2000)]
    return f(np.arange(len(rs))), np.percentile(bs, [2.5, 97.5]), Ce


lines = ["| group | n | estimator@B | charged FPE | F | 95% CI | strict (F<=1) | lenient (CI hi<=10) |", "|---|---|---|---|---|---|---|---|"]
for g in groups:
    rs = [r for r in R if r["name"].split("_")[1] == g]
    for k in keys:
        f, ci, Ce = F(rs, k)
        st = "yes" + (" (sig.)" if ci[1] <= 1 else "") if f <= 1 else "no"
        lines.append(f"| {g} | {len(rs)} | {k} | {Ce:.0f} | {f:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] | {st} | {'yes' if ci[1] <= 10 else 'no'} |")
open(os.path.join(HERE, "results.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
