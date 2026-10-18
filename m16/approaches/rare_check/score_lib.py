"""T350 scoring (DRI / T286 corrected rule): sampling's relative error sqrt((1-p)/(pC)) if pC >= 1, else 1 (zero
answer); win if ours < sampling - .05, loss if ours > sampling + .05, tie otherwise."""
import numpy as np


def case(est, p, C):
    r = abs(est / p - 1)
    s = np.sqrt((1 - p) / (p * C)) if p * C >= 1 else 1.0
    return dict(est=est, truth=p, ratio=est / p, r=r, s=s, C=C, pC=p * C,
                outcome="tie" if abs(r - s) < .05 else ("win" if r < s else "loss"))
