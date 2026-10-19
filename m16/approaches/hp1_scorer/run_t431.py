"""T431 runner (DRI pre-registration, TASKS T431 @ 791153c9): the T10 walk-quadrature recipe (and T391 isg, secondary)
exactly as frozen in T429 (claude/t429-walk-budget @ 79ecae63), on the sealed HP5 panel, one deterministic run per
model. Uses T429's own runner module (score.py afcb43325500: hash asserts, lazy-Sobol and compiled-walk swaps,
gated bitwise against T429's DEV3 rows in this container; see README). usage:
  PANEL=hp5 python run_t431.py RECIPE BUDGET      -> data/hp5/raw_t431_<recipe>_<budget>.jsonl (resumable)"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import score  # noqa: E402
assert score.PANEL == "hp5"
T429 = os.path.join(score.ROOT, "data/subs/t429_79ecae63/m16/approaches/walk_budget")
sys.path.insert(0, T429); os.chdir(T429)
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("t429_score", os.path.join(T429, "score.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)      # asserts the frozen hashes
recipe, B = sys.argv[1], int(float(sys.argv[2]))
E = S.load(recipe, "nb")
names, man, models, tr = score.load_panel()
out = os.path.join(score.D, f"raw_t431_{recipe}_{B}.jsonl")
done = {json.loads(l)["k"] for l in open(out)} if os.path.exists(out) else set()
for k, nm in enumerate(names):
    if k in done:
        continue
    t0 = time.process_time()
    e, c = E.est(models[k], B, None)
    with open(out, "a") as f:
        f.write(json.dumps(dict(k=k, name=nm, B=0, budget=B, recipe=recipe, est=e, cost=c,
                                cpu=time.process_time() - t0)) + "\n")
print("done", recipe, B, flush=True)
