"""External supervisor + isolated estimator worker. NOT submission code."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def worker(args):
    import flopscope as flops
    import flopscope.numpy as fnp
    import numpy as np
    from whestbench import SetupContext
    from whestbench.generation import sample_mlp
    from whestbench.seeds import derive_seed_streams

    manifest = json.loads((HERE / "sources.json").read_text())[args.method]
    path = HERE / "estimator_blocked.py" if manifest.get("derived") else Path(manifest["local_path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    spec = importlib.util.spec_from_file_location("t474_estimator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    est = module.Estimator()
    weight_ss, _, estimator_seed = derive_seed_streams(args.seed)
    mlp = sample_mlp(args.width, args.depth, rng=fnp.random.default_rng(weight_ss), seed=estimator_seed)
    model_hash = hashlib.sha256()
    for w in mlp.weights:
        model_hash.update(np.asarray(w).tobytes())
    setup_start = time.perf_counter()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as setup_meter:
        est.setup(SetupContext(width=args.width, depth=args.depth, flop_budget=2**41,
                              api_version="1.0", seed=474, submission_dir=str(path.parent)))
    setup_time = time.perf_counter()-setup_start
    start = time.perf_counter()
    with flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=120, quiet=True) as meter:
        pred = est.predict(mlp, 2**41)
    wall = time.perf_counter()-start
    pred = np.asarray(pred)
    finite = bool(np.isfinite(pred).all())
    shape_ok = pred.shape == (args.depth, args.width)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {
        "method": args.method, "width": args.width, "depth": args.depth,
        "mlp_seed": args.seed, "estimator_seed": estimator_seed,
        "source_sha256": manifest["sha256"], "weights_sha256": model_hash.hexdigest(),
        "versions": {k: importlib.metadata.version(k) for k in ["numpy", "flopscope", "whestbench"]},
        "setup_flops": setup_meter.flops_used, "setup_wall_seconds": setup_time,
        "flops": meter.flops_used, "cost_fraction": meter.flops_used / 2**41,
        "predict_wall_seconds": wall, "residual_seconds": meter.residual_wall_time_s,
        "finite": finite, "shape_ok": shape_ok,
        "local_contract_pass": finite and shape_ok and setup_time <= 5 and meter.residual_wall_time_s <= .4,
        "worker_cpu_seconds": usage.ru_utime + usage.ru_stime,
        "peak_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
    }
    np.save(args.output.with_suffix(".npy"), pred)
    args.output.write_text(json.dumps(result, indent=2)+"\n")


def supervise(args):
    import psutil
    if args.output.exists():
        raise RuntimeError(f"Refuse to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = {k:v for k,v in os.environ.items() if not re.match(r"^(V\d+_|X_)", k)}
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    cmd = [sys.executable, __file__, "--worker", "--method", args.method,
           "--width", str(args.width), "--depth", str(args.depth),
           "--seed", str(args.seed), "--output", str(args.output)]
    start, peak, cpu, reason = time.monotonic(), 0, 0, None
    with args.output.with_suffix(".log").open("w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        child = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                peak = max(peak, child.memory_info().rss)
                times = child.cpu_times()
                cpu = times.user + times.system
                if peak > 1_650_000_000:
                    reason = "RSS guard exceeded 1.65 GB"
                elif time.monotonic()-start > 150:
                    reason = "150 s worker wall cap"
                elif cpu > 120:
                    reason = "120 CPU-s worker cap"
                if reason:
                    # Only our direct Popen child; no process-name pattern.
                    proc.kill()
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(.02)
        code = proc.wait()
    result = json.loads(args.output.read_text()) if args.output.exists() else {}
    result.update(supervisor_peak_rss_bytes=peak, supervisor_cpu_seconds=cpu,
                  exit_code=code, stop_reason=reason,
                  total_wall_seconds=time.monotonic()-start)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(args.output.name, "pass", result.get("local_contract_pass"),
          "FLOPs", result.get("flops"), "RSS MB", round(peak/1e6),
          "wall", round(result["total_wall_seconds"], 2), "stop", reason, flush=True)
    if code or reason:
        raise SystemExit(code or 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--method", choices=list(json.loads((HERE/"sources.json").read_text())), required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    (worker if args.worker else supervise)(args)
