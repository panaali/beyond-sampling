"""T284: retrain panels (ARC scores over randomness in the weights = fresh retrains, zoo (16,10) config).
  dev2: the 80 models of the T234 40-vs-40 kernel check (2^30 zoo config, seeds 0-39, data seed 21; fused kernel and
        trainer.py), weights and truths OPEN -> dev2/weights.npz, dev2/manifest.json.
  hp2:  grid3's 2^30 zoo-config models, seeds 100-199 (fused kernel, data seed 41), from capacity_ceiling/hp2_run.py's
        state.pt (re-trained alone, bit-identical data ids) or grid3's state.pt once its step >= 2^20 -> SEALED
        panel hp2/weights.npz, hp2/manifest.json (truths only in hp2/sealed/truths.json).
  hp3:  T321 fresh retrains (seeds 200-299, data ids 2000-2099) from capacity_ceiling/hp3_run.py -> ../hp3/ (SEALED)
  hp4:  T328 fresh retrains (seeds 300-399, data ids 3000-3099) from capacity_ceiling/hp4_run.py -> ../hp4/ (SEALED)
  hp5:  T336 fresh retrains (seeds 400-499, data ids 4000-4099) from capacity_ceiling/hp5_run.py -> ../hp5/ (SEALED)
usage: python make_panel.py dev2 | python make_panel.py hp2|hp3|hp4|hp5 STATE_PT"""
import hashlib, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); CC = os.path.join(HERE, "..", "..", "capacity_ceiling")
which = sys.argv[1]; names, W, man = [], [], []
if which == "dev2":
    out = os.path.join(HERE, "dev2")
    for kern, pat in (("fused", "cmp_fused_s{}.npz"), ("torch", "cmp_torch_seed{}.npz")):
        for s in range(40):
            z = np.load(os.path.join(CC, "weights", pat.format(s))); names.append(f"{kern}_s{s}")
            W.append((z["Wi"], z["Wh"], z["Wo"]))
            man.append(dict(name=names[-1], trainer="fused.py (Triton)" if kern == "fused" else "trainer.py (torch)",
                            seed=s, n_train=2 ** 30, data_seed=21, config="zoo (16,10) defaults", source=pat.format(s)))
else:
    import torch as th
    out = HERE if which == "hp2" else os.path.join(HERE, "..", which); st = th.load(sys.argv[2], weights_only=False)
    spec = st["meta"]["spec"]
    assert st["step"] >= 2 ** 20, st["step"]
    for k, sp in enumerate(spec):
        if sp["name"].startswith("zoo_n30_s") and sp["n_train"] == 2 ** 30:
            assert sp["n_steps"] <= st["step"]
            names.append(f"{which}_s{sp['seed']}"); W.append((st["Wi"][k].double().numpy(), st["Wh"][k].double().numpy(),
                                                       st["Wo"][k][:10].double().numpy()))   # Wo padded to 16 rows
            man.append(dict(name=names[-1], trainer="fused.py (Triton)", seed=sp["seed"], n_train=2 ** 30, data_seed=41,
                            config="zoo (16,10) defaults", data_id=sp.get("data_id", sp.get("grid3_index", k))))
    assert len(names) == 100, len(names)
Wi = np.stack([w[0] for w in W]).astype(np.float64); Wh = np.stack([w[1] for w in W]).astype(np.float64)
Wo = np.stack([w[2] for w in W]).astype(np.float64)
os.makedirs(out, exist_ok=True); np.savez(os.path.join(out, "weights.npz"), Wi=Wi, Wh=Wh, Wo=Wo, names=np.array(names))
h = hashlib.sha256(Wi.tobytes() + Wh.tobytes() + Wo.tobytes()).hexdigest()
json.dump(dict(panel=which.upper(), n_models=len(names), weights="weights.npz", weights_sha256=h, models=man,
               truth=("truths.json (OPEN)" if which == "dev2" else "sealed/truths.json (SEALED)") +
               ": float64 MC, 1e8 per model, independent torch CUDA Philox stream per model"),
          open(os.path.join(out, "manifest.json"), "w"), indent=1)
print(which, len(names), "models, weights sha256", h)
