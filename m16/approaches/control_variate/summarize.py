"""Compute-adjusted MSE ratios from run_eval.py outputs. Ratio = (MSE_ref * evals_ref) / (MSE_est * evals_est).
srb/srb_cvA rows were run with a 1.6/sample charge; their evals are rescaled to the final 1.7/sample charge."""
import json, os, glob
HERE = os.path.dirname(os.path.abspath(__file__))
for lv in ("0", "0.0025", "0.005"):
    f = os.path.join(HERE, f"log_{lv}.txt")
    rows = [json.loads(l) for l in open(f) if l.startswith("{")]
    print(f"== sigma {lv}")
    for b in sorted({r["budget"] for r in rows}):
        R = {r["est"]: r for r in rows if r["budget"] == b}
        def cm(n):
            r = R[n]; e = r["evals"] * (1.7 / 1.6 if n.startswith("srb") else 1.0); return r["mse"] * e
        out = [f"budget {b}:"]
        for n in R:
            if n == "sampling": continue
            out.append(f"{n} x{cm('sampling')/cm(n):.2f} (vs strat x{cm('stratified')/cm(n):.2f}, bias {R[n]['bias']*100:+.3f}pp)")
        print("  " + " | ".join(out))
