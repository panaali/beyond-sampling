"""Loader for S1 checkpoints (T444).  from load_s1 import load; model, lang = load("dyck2_poshard_s10")
Each checkpoint stores cfg; a model whose every layer is plain hard with tol 1e-6 is also loadable by the
harness as attn='hard' (see README)."""
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mini_lm"))
from langs import LANGS  # noqa: E402
from s1models import S1Transformer  # noqa: E402


def load(name, models_dir=os.path.join(HERE, "models")):
    ck = torch.load(os.path.join(models_dir, name + ".pt"), weights_only=False)
    cfg = dict(ck["cfg"]); cfg.pop("arch")
    lang = LANGS[ck["lang"]]()
    model = S1Transformer(**cfg)
    model.load_state_dict(ck["state"])
    for b, beta in zip(model.blocks, ck.get("betas", [1.0] * len(model.blocks))):
        b.beta = beta
    model.eval()
    return model, lang


def names(models_dir=os.path.join(HERE, "models")):
    return sorted(f[:-3] for f in os.listdir(models_dir) if f.endswith(".pt"))
