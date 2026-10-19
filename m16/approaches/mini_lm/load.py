"""Loader for the T433 baselines.  from load import load; model, lang = load("dyck2_hard_s0")"""
import os
import torch

from langs import LANGS
from models import ReLURNN, TinyTransformer

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, models_dir=os.path.join(HERE, "models")):
    ck = torch.load(os.path.join(models_dir, name + ".pt"), weights_only=True)
    cfg = dict(ck["cfg"])
    lang = LANGS[ck["lang"]]()
    model = ReLURNN(**cfg) if cfg.pop("arch") == "rnn" else TinyTransformer(**cfg)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, lang


def names(models_dir=os.path.join(HERE, "models")):
    return sorted(f[:-3] for f in os.listdir(models_dir) if f.endswith(".pt"))
