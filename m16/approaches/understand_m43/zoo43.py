"""Weights of alg_zoo.zoo_2nd_argmax(4, 3, seed=s) as (Wi (4,), Wh (4,4), Wo (3,4)) numpy arrays."""
import numpy as np


def zoo_model(seed):
    import alg_zoo
    m = alg_zoo.zoo_2nd_argmax(4, 3, seed=seed)
    sd = {k: v.detach().numpy().astype(float) for k, v in m.state_dict().items()}
    return sd["rnn.weight_ih_l0"][:, 0], sd["rnn.weight_hh_l0"], sd["linear.weight"]
