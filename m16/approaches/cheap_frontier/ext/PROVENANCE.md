Copied inputs (not produced here):
- probe_v5_lb3_raw.jsonl: origin/claude/t241-neuron-account 44a3524a m16/approaches/probe_v5/results/lb3_raw.jsonl (T274a LB3 run).
- t274b_cone_bias.json: origin/claude/t218-tie-tail-labels 26e9291d m16/approaches/lenient_bb/results/cone_bias.json
  (T274b deductive root-cone baseline; paired bias approx - network on 400k samples per model, perfect integration).
- dev2/: origin/claude/t192-gpu-cost-law 65ee4726 m16/approaches/gpu_cost_law/hp2/dev2/{manifest.json,weights.npz} (T284 DEV2, 80 open retrains; official truths not yet published).
- probe_v5.py, t274b_cone_bias.py, t274b_frozen_bias.py: verbatim copies of probe_v5/v5.py and lenient_bb/{cone_bias,frozen_bias}.py at the commits above (imported for the DEV2 runs only).
