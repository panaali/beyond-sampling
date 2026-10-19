"""Width ladder scheduler for T392 (pre-registered in README): per k, rungs H in RUNGS, seed 0 first, seeds 1-2 only if
seed 0 reaches >= .95 (1e7 fresh inputs); a k stops at its first rung where all 3 seeds pass. Up to SLOTS runs at once,
lower rungs first; a k may run up to AHEAD rungs beyond its lowest undecided rung (speculative, does not change the
selection). Accuracies go to results/acc.json.

usage: python ladder.py [SLOTS] [AHEAD]
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KS, RUNGS, SEEDS, PASS = (1, 2, 5, 10), (16, 24, 32, 48, 64, 128, 192, 256), (0, 1, 2), 0.95
T, CHUNK = 10, 250_000


def load(tag):
    d = json.load(open(f"{HERE}/models/{tag}.json"))
    return d["k"], {k: np.asarray(v) for k, v in d["weights"].items()}


def forward(k, W, x):
    h = np.zeros((len(x), W["Whh"].shape[0]))
    for t in range(T // k):
        h = np.maximum(h @ W["Whh"].T + x[:, t * k:(t + 1) * k] @ W["Wih"].T, 0)
    return h @ W["Wout"].T


def accuracy(k, W, n=10_000_000, seed=392001):
    rng, ok = np.random.default_rng(seed), 0
    for _ in range(n // CHUNK):
        x = rng.standard_normal((CHUNK, T))
        ok += int((forward(k, W, x).argmax(1) == np.argsort(x, 1)[:, -2]).sum())
    return ok / n


def main(slots=4, ahead=1):
    accp = f"{HERE}/results/acc.json"
    acc = json.load(open(accp)) if os.path.exists(accp) else {}
    running = {}
    while True:
        for tag, p in list(running.items()):
            if p.poll() is not None:
                del running[tag]
                if os.path.exists(f"{HERE}/models/{tag}.json"):
                    a = accuracy(*load(tag))
                    acc[tag] = {"acc": a, "se": float(np.sqrt(a * (1 - a) / 1e7))}
                    json.dump(acc, open(accp, "w"), indent=1)
                    print(time.strftime("%H:%M"), tag, a, flush=True)
        jobs = []
        for k in KS:
            seen_undecided = 0
            for ri, H in enumerate(RUNGS):
                a = [acc.get(f"k{k}h{H}_s{s}", {}).get("acc") for s in SEEDS]
                if all(x is not None and x >= PASS for x in a):
                    break                                    # simplest passing rung for this k
                if any(x is not None and x < PASS for x in a):
                    continue                                 # rung failed
                if seen_undecided > ahead:
                    break
                seen_undecided += 1
                want = [0] if a[0] is None else list(SEEDS)
                jobs += [(ri, k, H, s) for s in want if f"k{k}h{H}_s{s}" not in acc]
        jobs = [j for j in sorted(jobs) if f"k{j[1]}h{j[2]}_s{j[3]}" not in running]
        if not jobs and not running:
            break
        for ri, k, H, s in jobs[:max(0, slots - len(running))]:
            tag = f"k{k}h{H}_s{s}"
            log = open(f"{HERE}/logs/{tag}.log", "w")
            running[tag] = subprocess.Popen(["bash", "-c", f"ulimit -v 2500000; exec {sys.executable} {HERE}/train.py {k} {H} {s}"],
                                            stdout=log, stderr=subprocess.STDOUT)
            print(time.strftime("%H:%M"), "start", tag, flush=True)
        time.sleep(30)
    print("ladder done", flush=True)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
