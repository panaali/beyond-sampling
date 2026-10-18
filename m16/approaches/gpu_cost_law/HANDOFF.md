# HANDOFF: T270, T234 grid3, T284 HP2/DEV2, T196 (8,5) audit (live state; updated at every check-in)

Last update: 2026-10-18 12:15 UTC. **T270 closed** (README record); t192lb stopped 12:13. Branch `claude/t192-gpu-cost-law` (never push to the shared branch; DRI/root merge).
DRI: session `session_01CD3g9Cvc3qk3pM11eyXXzj` (reply via claude-code-remote `send_message`). T281 scorer session:
`session_01UvVWREPpEYpUGudpzkioKd`. Budget: T234 + T270 <= 115 Colab units combined.

## Reaching the VMs
- CLI: `C=/home/user/colabcli-venv/bin/colab` (OAuth token in `~/.config/colab-cli/token.json`, container-local, not
  committed; if lost, redo the login in [COLAB.md](COLAB.md) and ask the owner for the code).
- `$C sessions`, `$C usage` (balance), `$C ls -s S content/...` (no leading slash), `$C download -s S /content/f local`,
  `$C upload -s S local /content/f`, `$C stop -s S`.
- **Do not use `colab exec` for polling**: each kernel is held by a blocking keep-alive exec, so a second exec returns
  nothing. Poll with `ls` / `download`. Start jobs by uploading a script to `/content/jobs/NAME.sh`: the keep-alive
  runner (`colab/keepalive.py`) renames it to `.started` within 20 s and runs it detached (`/content/jobs/NAME.sh.out`).

| VM | hardware | runs | started |
|---|---|---|---|
| t192f2 | A100 | T270 shard 0 resumed (`/content/t270_shard0.log`) + grid3 restarted (`/content/grid3.log`), job `colab/j_f2.sh` | 08:22 |
| (t192lb) | A100 | stopped 12:13 (shard 1 done 11:22; HP2/HP3/HP4 trained there) | |
| t192aud | CPU | T196 (8,5) audit seeds 1-4 (`/content/audit_{1..4}.log`, marker `AUDIT2DONE` in `/content/audit_done`) | 07:15 |
Stopped: t192t4 (07:05, queue finished); **t192fu reclaimed 07:23-08:12 despite the keep-alive** (grid3 lost, shard 0 kept 7 chunks); t192za was reclaimed ~06:20 (za pack lost, superseded by grid3).

## Local processes (this container; kill only these PIDs, never by pattern; AGENTS rule 9)
- Keep-alive loops (`colab/keepalive.sh S`, log `/tmp/claude-0/keepalive_S.log`): t192f2 (started 08:22), t192lb PID 3449,
  t192aud PID 1560. Kill the VM's loop BEFORE `colab stop`, or it keeps re-execing.
- Background sync: `colab/sync_f2.sh` (2 h loop from 08:25; restart when it ends): every 8 min pulls grid3's
  `state.pt` to `data/capacity_ceiling/grid3/state.pt` and T270 chunks from t192f2 + t192lb. **If t192f2 is reclaimed:**
  new A100 (`colab/newsession.sh NAME A100`), upload chunks (tar) + state.pt to `/content/cn/data/capacity_ceiling/grid3/`,
  rerun `colab/j_f2.sh`-style job (prefix6 resumes from chunks; fused_run resumes from state.pt).
  `capacity_ceiling/sync.sh S` pulls `/content/t234.tgz` (results only for grid3) into capacity_ceiling/.
A fresh session will not inherit these processes: restart a keep-alive per live VM with
`cd m16/approaches/gpu_cost_law && nohup ./colab/keepalive.sh S >/dev/null 2>&1 &` (one per VM, check none is running).

## T270 (all k = 6 prefix region masses of M_16,10)
- Running: `python prefix6.py SHARD 2 11 40` (via `colab/q_t270.sh`), 3,811 subtrees, shard s takes i = s mod 2;
  chunk files `results/prefix6/shard{s}_chunk{c:04d}.npz` every 40 subtrees, `roots.npz` (root masses, sum
  0.999999983583). Log ends with `T270DONE`. Resumable: rerunning the same command skips positions already in chunks.
- Sync + bounds (run each ~10% of coverage):
  `colab/sync_t270.sh t192fu t192lb && python prefix6_bounds.py` -> `results/prefix6/bounds.json`.
  Chunk npz files are git-ignored (~4 MB each, ~800 MB total); commit bounds.json, logs, roots.npz only.
- **Early-stop rule (DRI):** stop both shards if the lower bound on g_1e-7 (= N6(1-1e-7)/N5(1-1e-7)) is >= 15; then
  spend the freed units raising grid3-style 2^34 runs to 100 seeds. Otherwise run to completion and report
  g_delta(5->6) for 1e-5, 1e-6, 1e-7, the mass-rank slope and leaf sums.
- **ROOT CAUSE (10:15, cx6.py):** estT geometry defect in some k = 6 cells: stored rays miss true vertices and the pulling
  triangulation drops volume -> masses are UNDERestimates; integration rules are fine (sx6.py); roots verified by MC
  (mc6.py). Fix: `remass6.py CHUNK [workers] cuda` (pass 1 estT; short subtrees re-walked with qhull rays + Delaunay)
  -> `results/prefix6/remass_<chunk>.npz`, used by prefix6_bounds.py in place of the chunk; bounds use U + D (D = exact
  per-chunk shortfall, `worst_chunks` in bounds.json) and are PROVISIONAL. DRI plan: validate on one chunk (leaf =
  root to ~1e-9), remass in descending D until D << 1e-7; report g_N(5->6) at shard 1's end. estT bug = backlog T316
  (propose a patch, don't edit estT). Test running 10:20: `/content/remass_test.log` on t192f2 (positions 125, 81).
- **Memory:** shard 0's main process on t192f2 grew ~20 GB/h (33 GB at 10:26); watchdog `colab/wd_shard0.sh` (job on
  t192f2) kills + restarts it right after a chunk is written when RSS > 22 GB (log lines `WD ...`; the old wrapper's
  `T270DONE` after a kill is bogus; real end = `T270DONE_REAL`). Shard 1 on t192lb is fine (11.6 GB).
- (history) **OPEN ISSUE (07:25): k = 6 leaf masses sum below their root masses** (per-chunk deficit up to 4.35e-5, total 5.1e-5
  over 520 subtrees; k = 5 was exact to 1.4e-9). Bounds at delta <= 1e-5 are invalid until fixed (processed + U <
  1 - delta saturates them). Diagnosis: `python diag6.py results/prefix6/shard1_chunk0001.npz 2` (per-subtree leaf vs
  root; ran on t192f2 with `DIAG_DEV=cuda`, local RAM too small) -> `/content/diag6.log` on t192f2. **Found 09:00:** the
  chunk's deficit is one subtree, position 125 (leaf 0.0031237 vs root 0.0031672, -4.35e-5 = 1.4%), with a region of mass
  exactly 0; 12 other subtrees agree to ~1e-10. Independent MC of that subtree: `mc6.py 125 1e8` on t192f2
  (`/content/mc6_125.log`, json `results/prefix6/mc6_125.json` there): unmatched_mc > 0 = lost geometry; MC mass in the
  zero-mass region = mass rule failure. DRI pre-approved: if masses must be recomputed, stop shard 0, keep region counts.
- Progress 07:23: shard 0 280/1906 at 3624 s (slowed by grid3: ~13:50 finish), shard 1 240/1905 at 2033 s (~11:20).
- Progress 07:05: shard 0 200/1906 subtrees at 2475 s (~6.5 h/shard -> finish ~13:00 / ~13:20 UTC); shard 1 no chunk
  yet. Bounds after 4 chunks: g_1e-7 >= 3.5, g_1e-6 >= 4.7 (no early stop).

## T234 grid3 (capacity_ceiling, 1025 models, fused kernel)
- `capacity_ceiling/q_grid3.sh` (uploaded as a job on t192fu): `fused_run.py grid3 --data-seed 41 --eval kernel
  --eval-n 1e8 --chunk 8192`, seeds 100-199 x {2^30..2^33 zoo, 2^31 lr x0.5 / x2 / warm-up x4 / 3 hard-case mixes},
  2^34 x seeds 100-124 (contingency because T270 runs long). Checkpoint `/content/cn/data/capacity_ceiling/grid3/state.pt`
  (step count in `st["step"]`; ~6.2 ms/step). Restarted 08:24 on t192f2; expected end ~13:30 UTC (critical path: 2^34 = 16.7M
  steps). Output `results/grid3.json`, `weights/grid3_*.npz`, `/content/grid3.done`; archive `/content/t234.tgz`.
- Analysis pending: per config mean, sd, top-5 mean, max with bootstrap CIs, plus error anatomy -> capacity_ceiling README.

## T284 panels (gpu_cost_law/hp2/)
- DEV2 (OPEN): DONE. `hp2/dev2/weights.npz` (sha256 22d1c7c2...343bec), `truths.json` (sha256 582c21cf...a9785;
  1e8/model; min .81355 max .96023 mean .92012 sd .03163), sent to DRI 07:27.
- HP2 (SEALED): DONE 08:47 (`capacity_ceiling/hp2_run.py` + `fused_gid.py`, grid3 data ids; delivered to the scorer;
  weights file sha256 41709001..., truths sha256 1a6a44f4...). Pending: bit-identity check vs grid3's copies at step >= 2^20. Original plan:: download state.pt, `python hp2/make_panel.py hp2 STATE_PT`
  (local), upload `hp2/weights.npz` + run `truth_panel.py hp2 1e8` on a GPU VM -> `hp2/sealed/truths.json` ONLY.
  Send the scorer (T281) path + weights/truth sha256; send DRI hashes + min/max/mean/sd. Never publish per-model truths.
- HP3 (T321) delivered 11:20, HP4 (T328) delivered 12:12 (`capacity_ceiling/hp3_run.py`, `hp4_run.py`; data ids
  2000-2099 / 3000-3099; truths sealed in `hp3/sealed/`, `hp4/sealed/`).
- HP1 truths likewise stay only in `hp1/sealed/truths.json`.

## T196 (8,5) audit
Seed 0: charged 2.177e11, complete 2.685e11, ratio 1.23 (`results/audit85/audit_0.log`). Seed 1: ratio 1.22 (`audit_1.log`). Seeds 2-4 running; relaunched 07:15
(`colab/q_audit2.sh`). When done: download `/content/audit_{1..4}.log` to `results/audit85/`, add to README, stop t192aud.

## Units
304.80 at login; 197.41 at 12:13 (one A100 left, 5.4 u/h) (~10.7 u/h now: 2 A100 at 5.3 + CPU 0.08). T234 + T270 spend so far ~34; projected
~100 of 115 at T270 completion. Stop each A100 as soon as its shard (and, on fu, grid3) is done and synced.
