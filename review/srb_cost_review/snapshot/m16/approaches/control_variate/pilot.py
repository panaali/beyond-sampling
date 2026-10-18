"""Neyman allocation for prefix strata from a pilot on the ORIGINAL model only (fixed design, reused for all copies)."""
import sys, json, numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16")
from common import Wi, Wh, Wo
import cv
P = cv.pc()["P"]; rng = np.random.default_rng(55)
out = {}
sd = []
for h in range(cv.NS):
    x = cv.draw(h, 20000, rng); q = cv.Q((Wi, Wh, Wo), x)
    B = cv.basis9(x, "A") - cv.pc()["MU_A"][h]
    c = np.linalg.lstsq(np.hstack([np.ones((len(x),1)),B]), q, rcond=None)[0]
    r = q - np.hstack([np.ones((len(x),1)),B]) @ c
    sd.append((q.std(), r.std()))
sd = np.array(sd); print(np.round(sd, 4))
for j, name in enumerate(("plain", "cv")):
    a = P * sd[:, j]; a = 0.5 * a / a.sum() + 0.5 * P   # half Neyman, half proportional (robustness to noise)
    out[name] = list(a)
json.dump(out, open("alloc.json", "w"), indent=1); print(out)
