"""Save a grid model in both formats and register its accuracy (T421)."""
import hashlib
import json
import os

import numpy as np
import torch as th

from evaluate import accuracy

HERE = os.path.dirname(os.path.abspath(__file__))


def save_model(tag, sd, *, H, T, seed, stream, provenance):
    """sd: alg_zoo DistRNN state_dict (rnn.weight_ih_l0 (H,1), rnn.weight_hh_l0 (H,H), linear.weight (T,H))."""
    sd = {k: v.detach().cpu().float() for k, v in sd.items()}
    assert sd["rnn.weight_hh_l0"].shape == (H, H) and sd["linear.weight"].shape == (T, H)
    pth = os.path.join(HERE, "weights", f"{tag}.pth")
    th.save(sd, pth)
    Wi = sd["rnn.weight_ih_l0"].numpy()[:, 0].astype(np.float64)
    Wh = sd["rnn.weight_hh_l0"].numpy().astype(np.float64)
    Wo = sd["linear.weight"].numpy().astype(np.float64)
    np.savez(os.path.join(HERE, "weights", f"{tag}.npz"), Wi=Wi, Wh=Wh, Wo=Wo)
    rec = {"tag": tag, "H": H, "T": T, "seed": seed, "stream": stream,
           "pth": f"weights/{tag}.pth", "npz": f"weights/{tag}.npz",
           "sha256_npz_weights": hashlib.sha256(np.concatenate([Wi, Wh.ravel(), Wo.ravel()]).tobytes()).hexdigest(),
           **accuracy(Wi, Wh, Wo), "provenance": provenance}
    json.dump(rec, open(os.path.join(HERE, "results", "models", f"{tag}.json"), "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("tag", "acc", "se")}), flush=True)
    return rec
