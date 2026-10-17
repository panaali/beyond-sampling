from pathlib import Path
import numpy as np
import hashlib
def frozen_models():
    zero = (np.zeros(4), np.zeros((4, 4)), np.zeros((3, 4)))
    positive = (np.ones(4), np.ones((4, 4)) / 8, np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]))
    models = [('zero', zero, None), ('positive', positive, None)]
    import torch
    for k, seed in ((30, 0), (30, 1), (30, 4), (24, 4)):
        path = Path('/workspace/work/t114_checkpoints') / f'2nd_argmax_4_3_{2 ** k}_{seed}.pth'
        sd = torch.load(path, map_location='cpu', weights_only=True)
        model = (sd['rnn.weight_ih_l0'][:, 0].double().numpy().copy(), sd['rnn.weight_hh_l0'].double().numpy().copy(), sd['linear.weight'].double().numpy().copy())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        models.append((f'k{k}_s{seed}_clean', model, digest))
        rng = np.random.default_rng(124)
        noisy = tuple((a + 0.05 * rng.standard_normal(a.shape) for a in model))
        models.append((f'k{k}_s{seed}_abs05_n124_0', noisy, digest))
    hidden=models[1][1]
    models.append(("constant_zero_readout",(hidden[0],hidden[1],np.zeros((3,4))),None))
    models.append(("constant_duplicate_readout",(hidden[0],hidden[1],np.array([[1.,1.,1.,1.],[1.,1.,1.,1.],[0.,0.,0.,0.]])),None))
    return models
