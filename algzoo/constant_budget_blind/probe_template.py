"""probe.py: the ONLY interface an investigator has to the model (T24, frozen before any run).

Call hidden(x) and predict(x). Do not read or print anything else from this module, and do not
import alg_zoo or torch in your own scripts.
"""
import numpy as np

H = __H__          # number of hidden neurons
T = 10             # sequence length

def _load():
    import torch
    from alg_zoo import zoo_2nd_argmax
    torch.set_num_threads(1)
    return torch, zoo_2nd_argmax(H, T).eval()

_torch, _model = _load()


def hidden(x):
    """x: (N, 10) array of real inputs. Returns the final hidden state h_10, shape (N, H), float64."""
    x = np.asarray(x, dtype=np.float32).reshape(-1, T)
    out = []
    with _torch.no_grad():
        for i in range(0, len(x), 50_000):
            _, h = _model.rnn(_torch.tensor(x[i:i + 50_000]).unsqueeze(-1))
            out.append(h[0].numpy().astype(np.float64))
    return np.concatenate(out) if out else np.zeros((0, H))


def predict(x):
    """x: (N, 10). Returns the model's predicted position (0..9) of the 2nd-largest input, shape (N,)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1, T)
    out = []
    with _torch.no_grad():
        for i in range(0, len(x), 50_000):
            out.append(_model(_torch.tensor(x[i:i + 50_000])).numpy().argmax(-1))
    return np.concatenate(out) if out else np.zeros(0, dtype=int)
