"""D8 (T380 development, DEV models only): variance over 16 alternative fixed rules, T10 frozen vs cv split, 64k and 16k.
Both are unbiased over the rule randomisation, so variance gain = MSE gain (up to the bias of a single rule)."""
import importlib.util, sys
import numpy as np
sys.path.insert(0, "../..")
import leaderboard as LB
import estimator as E0
spec = importlib.util.spec_from_file_location("cvE", "cv/estimator.py"); E1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(E1)
for T in (4, 5):
    ms = LB.models(16, T, dev=True)
    for name in list(ms)[2::4]:
        m = ms[name]; out = []
        for b in (16000, 64000):
            r0 = [E0.est(m, b, rule_seed=5000 + 17 * k) for k in range(16)]
            r1 = [E1.est(m, b, rule_seed=5000 + 17 * k) for k in range(16)]
            v0 = np.var([e for e, _ in r0]) * np.mean([c for _, c in r0]); v1 = np.var([e for e, _ in r1]) * np.mean([c for _, c in r1])
            out.append(f"{b}: var*cost ratio frozen/cv {v0/v1:.2f} (means {np.mean([e for e,_ in r0]):.5f} {np.mean([e for e,_ in r1]):.5f})")
        print(f"T{T} {name}: " + " | ".join(out), flush=True)
