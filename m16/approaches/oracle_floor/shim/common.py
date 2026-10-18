"""Shim for m16/common.py (torch and alg_zoo are not installed in this container): the target M_16,10 weights
from claude/t241-neuron-account neuron_account/m16_weights.npz, copied to data/oracle_floor/target_weights.npz."""
import os
import numpy as np
_z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../data/oracle_floor/target_weights.npz"))
Wi, Wh, Wo = (_z[k].astype(np.float64) for k in ("Wi", "Wh", "Wo"))
H, T = Wh.shape[0], Wo.shape[0]
