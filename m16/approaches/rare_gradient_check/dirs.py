"""Pre-registered directions (T385). All directions are vectors in the 432-dim weight space (core.flat order).
- neuron gain i: d = w restricted to neuron i's incoming weights (Wi[i], Wh[i, :]); d log P / d eps of
  P(W + eps d) is the gain elasticity E(i) (scaling the row by 1 + eps scales h_i by 1 + eps: ReLU homogeneity).
- interventions (orig): d = w restricted to a fixed set of weights, i.e. scaling those weights by 1 + eps.
- random: unit Gaussian directions, numpy default_rng(385000 + 10 * model index + j)."""
import numpy as np

import core

MODELS = ["orig", "dev3_s528"]
INTERV = {"coupling n6<->n7": [(6, 7), (7, 6)], "n2->n6": [(6, 2)], "n2->n7": [(7, 2)],
          "n2->n6 & n2->n7 (joint)": [(6, 2), (7, 2)], "n6<-n7 only": [(6, 7)], "n7<-n6 only": [(7, 6)]}


def wh_index(i, j):
    return 16 + 16 * i + j


def neuron_dir(w, i):
    d = np.zeros(432); idx = [i] + [wh_index(i, j) for j in range(16)]; d[idx] = w[idx]; return d


def interv_dir(w, pairs):
    d = np.zeros(432); idx = [wh_index(i, j) for i, j in pairs]; d[idx] = w[idx]; return d


def all_dirs(nm):
    w = core.flat(core.load(nm)); names, D = [], []
    for i in range(16):
        names.append(f"gain n{i}"); D.append(neuron_dir(w, i))
    mi = MODELS.index(nm)
    for j in range(3):
        u = np.random.default_rng(385000 + 10 * mi + j).standard_normal(432); names.append(f"random {j}"); D.append(u / np.linalg.norm(u))
    if nm == "orig":
        for k, v in INTERV.items():
            names.append("interv " + k); D.append(interv_dir(w, v))
    return names, np.array(D)
