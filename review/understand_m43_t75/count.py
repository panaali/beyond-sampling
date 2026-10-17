"""T84 item 3: automatic op count. Every scalar operation the re-implemented recipe (recipe.py) executes on
model-dependent numbers is counted by the `Num` type: + - * / (1 each), abs, comparisons (1 each), sqrt and
atan2 (1 each). The same counter runs one plain forward pass + label + correctness check for sampling, so both sides
are counted under one convention (FLOPs incl. comparisons; ARC's 72 is multiply-adds only).

usage: python count.py [n_models_per_seed=4]  -> count.log, count.json
"""
import json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "understand_m43"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "m16", "approaches", "arc_m43_repro"))
import numpy as np
import recipe as R
from mine import zoo_w, noisy_w

C = {"add": 0, "mul": 0, "div": 0, "abs": 0, "cmp": 0, "sqrt": 0, "atan2": 0}


def v(x):
    return x.x if isinstance(x, Num) else x


class Num:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = float(x)

    def _b(self, o, f, kind):
        C[kind] += 1; return Num(f(self.x, v(o)))

    def __add__(self, o): return self._b(o, lambda a, b: a + b, "add")
    def __radd__(self, o): return self._b(o, lambda a, b: b + a, "add")
    def __sub__(self, o): return self._b(o, lambda a, b: a - b, "add")
    def __rsub__(self, o): return self._b(o, lambda a, b: b - a, "add")
    def __mul__(self, o): return self._b(o, lambda a, b: a * b, "mul")
    def __rmul__(self, o): return self._b(o, lambda a, b: b * a, "mul")
    def __truediv__(self, o): return self._b(o, lambda a, b: a / b, "div")
    def __rtruediv__(self, o): return self._b(o, lambda a, b: b / a, "div")
    def __pow__(self, p): C["mul"] += 1; return Num(self.x ** p)
    def __neg__(self): return Num(-self.x)
    def __abs__(self): C["abs"] += 1; return Num(abs(self.x))
    def __lt__(self, o): C["cmp"] += 1; return self.x < v(o)
    def __le__(self, o): C["cmp"] += 1; return self.x <= v(o)
    def __gt__(self, o): C["cmp"] += 1; return self.x > v(o)
    def __ge__(self, o): C["cmp"] += 1; return self.x >= v(o)
    def __float__(self): return self.x


def nsqrt(a): C["sqrt"] += 1; return Num(math.sqrt(v(a)))
def natan2(a, b): C["atan2"] += 1; return Num(math.atan2(v(a), v(b)))


def wrap(model):
    Wi, Wh, Wo = R.as_lists(model)
    return [Num(x) for x in Wi], [[Num(x) for x in r] for r in Wh], [[Num(x) for x in r] for r in Wo]


def counted_recipe(model):
    R.SQRT, R.ATAN2 = nsqrt, natan2
    for k in C: C[k] = 0
    ch = R.Charge()
    # patch module-level default args (sqrt/atan2 are bound at def time)
    R.clip.__defaults__ = (nsqrt,); R.unit.__defaults__ = (nsqrt,); R.frac.__defaults__ = (natan2,)
    try:
        est = v(R.recipe(wrap(model), ch))
    finally:
        R.SQRT, R.ATAN2 = math.sqrt, math.atan2
        R.clip.__defaults__ = (math.sqrt,); R.unit.__defaults__ = (math.sqrt,); R.frac.__defaults__ = (math.atan2,)
    return est, dict(C), ch.n


def counted_sample(model, x):
    """One sample: 3 RNN steps (h0 = 0, so step 1 is x0*Wi), ReLU, readout, argmax, 2nd-argmax label, compare."""
    for k in C: C[k] = 0
    Wi, Wh, Wo = wrap(model)
    xs = [Num(t) for t in x]
    zero = Num(0.0)
    h = [xs[0] * Wi[k] for k in range(4)]
    h = [a if a > zero else zero for a in h]
    for t in (1, 2):
        pre = [sum((Wh[k][j] * h[j] for j in range(4)), xs[t] * Wi[k]) for k in range(4)]
        h = [a if a > zero else zero for a in pre]
    lo = [sum((Wo[o][j] * h[j] for j in range(1, 4)), Wo[o][0] * h[0]) for o in range(3)]
    am = 0
    for o in (1, 2):
        if lo[o] > lo[am]:
            am = o
    a, b, c = xs                      # 2nd-largest of three inputs: 3 comparisons
    _ = (a > b, b > c, a > c)
    return dict(C)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    rows = []
    t0 = time.time()
    for zs in range(5):
        base = zoo_w(zs)
        for m in [base] + noisy_w(base, n, 0.05, 31):
            est, cnt, charged = counted_recipe(m)
            fl = sum(cnt.values())
            rows.append(dict(zoo=zs, est=est, counts=cnt, flops=fl, charged_t75_conv=charged))
            print(f"zoo {zs}: est {est:.5f}  FLOPs {fl:7d}  ({cnt})  T75-convention charge {charged}", flush=True)
    s = counted_sample(zoo_w(0), [0.3, -1.2, 0.7]); sf = sum(s.values())
    print(f"one sample (forward + label + check): FLOPs {sf} {s}")
    fl = np.array([r["flops"] for r in rows]); chg = np.array([r["charged_t75_conv"] for r in rows])
    print(f"recipe FLOPs / T75-convention charge: mean {np.mean(fl / chg):.3f} (range {np.min(fl / chg):.3f}-{np.max(fl / chg):.3f})")
    print(f"recipe FLOPs in sampling-pass units: mean {np.mean(fl) / sf:.0f} samples; T75 charge / 72: {np.mean(chg) / 72:.0f} samples")
    print(f"=> MSE-ratio multiplier from the all-FLOPs convention vs T75's (charge vs 72): "
          f"{(np.mean(fl) / sf) / (np.mean(chg) / 72):.3f}")
    json.dump(dict(rows=rows, sample=s, sample_flops=sf), open(os.path.join(HERE, "count.json"), "w"), indent=1)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
