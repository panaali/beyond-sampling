# gpu_cost_law: T192 (compiled/GPU exact reference and cost law; T5 go/no-go)
**Question.** What is the real cost law of exact enumeration and recipe108 (estT_fix) along T at H = 8, and is an exact
M_16,10 answer key (T5) reachable with the owner's Colab budget (~305 units)?
**Method.** Colab is driven from this container with the official `google-colab-cli` (OAuth, `colab new/upload/exec`).
Code goes up as a tarball of `m16/` (repo is private; no token on the VM); long runs are detached (`nohup`) and polled
because `colab exec` replies time out. Stage S0: unchanged Python on Colab CPU. Next: compiled port (numba, batched
float64 masses), validated against E46/E48 before any new rung. Scripts in `colab/`, outputs in `results/`.
**Status.** running (S0 done; (8,5) seeds 1-4 baseline with unchanged code running on Colab CPU).
## Results
- S0 [measured]: Colab CPU reproduces (8,4) zoo0 exact (ops 1100365022, cells 8455 identical; acc diff 7.9e-12) and
  recipe108 (ops 886579830, cells 6182 identical; acc diff 7.1e-12). 48 s / 45 s per run. `colab/s0.py` ->
  `results/s0_colab_cpu_84_zoo0.json`.
## Compute accounting
Colab units: balance 304.80 at start (2026-10-17). CPU runtime 0.08 units/h. Ops = `m43.Ops` counter (unchanged).
## Claims
- [measured] Unchanged code is portable to Colab CPU at <=1e-11 accuracy difference with identical op/cell counts (1 model).
## Alternatives not ruled out
The 8e-12 difference is attributed to numpy/libm version; not isolated.
## Failure modes
Colab session reclaim kills detached runs (partials lost unless written incrementally).
