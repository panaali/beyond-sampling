"""One loader for every model S4 scores: T433 baselines, MLP1 (pass models_dir) and the S1 (T444) group, read from a
detached checkout of claude/t444-mini-lm-s1 e3a012cf at S1_DIR (names dyck2_D6_s*, dyck2_R3a_s*)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mini_lm"))
S1_DIR = os.environ.get("S1_DIR", "/home/user/wt-s1/m16/approaches/mini_lm_s1")
S1_NAMES = [f"dyck2_{v}_s{s}" for v in ("D6", "R3a") for s in (10, 11, 12)]


def load_any(name, models_dir=None):
    if name in S1_NAMES:
        sys.path.insert(0, S1_DIR)
        from load_s1 import load as load1
        return load1(name, models_dir=os.path.join(S1_DIR, "models"))
    from load import load
    return load(name) if models_dir is None else load(name, models_dir=models_dir)
