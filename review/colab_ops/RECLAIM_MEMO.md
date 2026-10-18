# Colab GPU reclaims under `colab exec` + `nohup` (2026-10-18)

D = documented, I = inferred; [n] = sources below.

## 1. Documented termination reasons
- **Idle:** "Runtimes will time out if you are idle"; "Idle timeouts only apply if code execution terminates". Limits "vary over time" and are not published [1] (D). Google's VS Code extension source: an idle runtime "is reclaimed within ~30 minutes and none outlive 24 hours" [2] (D).
- **Activity without a browser:** the CLI dropped its keep-alive ping on 2026-09-25 (v0.7.3+) as "effectively a no-op". The backend now keeps runtimes alive "based on kernel activities and active connections" [3][4] (D). A repo contributor (affiliation unverified): Colab "mostly depends on checking for live connections". After a disconnect, the VM "will likely be shut down soon after" [5].
- **Lifetime:** "at most 12 hours". Pro+ gets "continuous code execution for up to 24 hours" [1] (D).
- **Background execution:** Pro+ [6] and Google AI Ultra (since 2026-09-22) [7] (D). Pay As You Go and Pro don't get it (I).
- **Compute units/abuse:** runtimes end when the balance is exhausted. The SSH and automation restrictions apply only to the free tier without a balance [1] (D).
- **Preemption:** Colab "prioritizes users who are actively programming" [1] (D). The actual GPU preemption policy isn't public.

## 2. Likely explanation (I)
With `nohup`, the cell returns at once. After that the kernel is idle and no connection exists, except a few seconds per poll. In a third-party test, a CPU VM with no connection survived 3 h; GPUs were untested [8]. So idle reclaim (about 30 min) looks strict on GPUs and lax on CPUs, though this isn't documented per accelerator. Capacity preemption of idle premium GPUs fits equally well.

Rule out:
- A CLI older than 0.7.4 falsely reports "lost (404/401)" at about 60 min because the proxy token expired [9]. Check `colab version`, then `colab sessions`.
- The job exhausted host RAM or disk.
- Compute units ran out (check `colab usage`).

## 3. Mitigations
- **CLI keepalive:** none; the old ping is a no-op [3] (D).
- **Real activity (I):**
  - Run the job inside the cell: launch `colab exec --timeout 86400 -f run.py` in the background locally, where `run.py` blocks on `subprocess.run`. Poll via `colab download`/`ssh`, since `exec` queues on a busy kernel.
  - And/or hold a `colab ssh` connection open (ServerAliveInterval, reconnect loop), as recommended in [5].
  - Test: two 1-h L4 runs, `nohup sleep` against a held connection.
- **Tier:** Pro+ ($49.99/mo) or Ultra. Whether their background execution covers the CLI is undocumented, and Ultra's page says "Only available on the web" [10].
- **Checkpointing:** save weights, optimizer, RNG state and chunk index every 5-10 min to GCS (`colab auth`), Drive, or via `colab download`. Make chunks resumable.
- **Same-account alternatives (each needs a GCP project, billing and GPU quota):**
  - *Colab Enterprise:* idle shutdown 10 min to 24 h, or off (`0s`) [11]. Headless runs with `gcloud colab executions create` (24 h timeout) [12]. L4 $0.67 + $0.11 fee per hour; A100-40GB $3.52 + $0.59 per hour; plus vCPU/RAM [13].
  - *GCE VM:* g2-standard-8 (L4) about $0.85/h, a2-highgpu-1g (A100) about $3.67/h [14]. No idle reclaim, but stop it yourself. Use a Deep Learning VM image and `gcloud compute ssh`.
  - *Colab "custom GCE VM":* deprecated 2025-03-21 [15]. The CLI only allocates managed runtimes anyway (I, from the source).

[1] https://research.google.com/colaboratory/faq.html
[2] https://github.com/googlecolab/colab-vscode/blob/main/src/jupyter/storage.ts
[3] https://github.com/googlecolab/google-colab-cli/pull/144
[4] https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md
[5] https://github.com/googlecolab/google-colab-cli/issues/160
[6] https://workspaceupdates.googleblog.com/2024/06/google-workspace-colab-pro-and-colab-pro-plus.html
[7] https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/
[8] https://github.com/danbarua/mighty-colab/pull/82
[9] https://github.com/googlecolab/google-colab-cli/issues/106
[10] https://support.google.com/googleone/answer/16286513
[11] https://docs.cloud.google.com/colab/docs/idle-shutdown
[12] https://docs.cloud.google.com/sdk/gcloud/reference/colab/executions/create
[13] https://cloud.google.com/colab/pricing
[14] https://cloud.google.com/products/compute/pricing/accelerator-optimized
[15] https://research.google.com/colaboratory/marketplace.html
