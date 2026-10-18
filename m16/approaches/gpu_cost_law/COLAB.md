# Running on the owner's Google Colab from an agent container (T192 recipe)

Written by the T192 worker, 2026-10-18. Works from a headless cloud container; nothing runs on the owner's machine.

## 1. Install and log in (once per container, needs the owner for ~1 minute)
```bash
python3 -m venv /home/user/colabcli-venv
/home/user/colabcli-venv/bin/pip install -q google-colab-cli        # official CLI (tested 0.7.4)
C=/home/user/colabcli-venv/bin/colab
```
The CLI's login is a remote copy-paste OAuth flow that blocks on `input()`. Run it detached with a FIFO as stdin so the
agent can feed the code back later:
```bash
D=/tmp/colabauth; mkdir -p $D; rm -f $D/in; mkfifo $D/in
(setsid nohup bash -c "exec 3<>$D/in; $C usage <&3 > $D/out.log 2>&1" >/dev/null 2>&1 &)
sleep 5; cat $D/out.log          # prints an accounts.google.com URL
```
Send the URL to the owner. They sign in with the Google account that holds the compute units and approve; Google's
page (sdk.cloud.google.com) shows an authorization code. They paste it to the agent, which feeds it in:
```bash
printf '%s\n' '<code from owner>' > $D/in; sleep 5; cat $D/out.log   # -> "Current balance: ... compute units"
```
The refresh token lands in `~/.config/colab-cli/token.json` (container-local; it disappears with the container).
The code is single-use: every new container needs a new URL + code from the owner. Scopes: profile, email,
cloud-platform, colaboratory, drive.file. The owner can revoke at https://myaccount.google.com/permissions.

## 2. Pick the cheapest runtime that fits (rates measured 2026-10-18, Colab Pro)
| runtime | `colab new` flag | vCPU | RAM | GPU | units/h |
|---|---|---|---|---|---|
| CPU | (none) | 2 | 12 GB | - | 0.08 |
| T4 | `--gpu T4` | 2 | 12 GB | T4 16 GB (fp64 weak) | ~1.1 |
| L4 | `--gpu L4` | 12 | 52 GB | L4 23 GB (fp64 weak) | ~1.5 |
| A100 | `--gpu A100` | 12 | 83 GB | A100 40 GB (fp64 9.7 TF) | ~5.3 |
`colab usage` shows balance and the summed rate of all live sessions (rate per runtime = difference before/after
`colab new`). Rules we follow: test on the local CPU first, dry-run on CPU/L4, use the A100 only for float64-heavy GPU
work it can keep busy, and `colab stop -s NAME` as soon as a queue ends (idle sessions keep billing).

## 3. Code in, runs, results out
The repo is private and the VM has no GitHub token: upload a tarball instead.
```bash
cd /home/user/clean-neurons
tar czf /tmp/code.tgz --exclude=results m16/*.py m16/truth.txt m16/approaches/{estT_fix,understand_m43,arc_m43_repro,gpu_cost_law}
$C new -s myrun --gpu L4
$C upload -s myrun /tmp/code.tgz /content/code.tgz
printf 'import subprocess as s\nprint(s.run("cd /content && mkdir -p cn && tar xzf code.tgz -C cn && pip install -q numba git+https://github.com/alignment-research-center/alg-zoo",shell=True,capture_output=True,text=True).stdout[-500:])\n' | $C exec -s myrun
```
`colab exec` gives up waiting after about a minute, so start long jobs detached and poll their logs:
```bash
printf 'import subprocess\nsubprocess.Popen("cd /content && nohup bash queue.sh > queue.out 2>&1 &",shell=True)\n' | $C exec -s myrun
printf 'import subprocess as s\nprint(s.run("tail -n 5 /content/queue.out",shell=True,capture_output=True,text=True).stdout)\n' | $C exec -s myrun
```
Results: `colab download -s myrun /content/res.tgz local.tgz` (see `colab/sync.sh`). Large uploads (>~100 MB) can
fail through the agent proxy; generate big data on the VM instead. Write per-item logs (one JSON line per finished
unit) so a reclaimed session can resume.

## 4. Pitfalls met
- In bash queue scripts use `local` variables inside functions (a global `s=$(date)` overwrote a loop variable).
- `pgrep -f pattern` matches its own shell; fine on a private VM, never on the shared sandbox (AGENTS.md rule 9).
- Fork the multiprocessing pool before the parent touches CUDA; workers must not use CUDA.
- Bound in-flight results when workers produce data faster than the GPU consumer (RAM).
