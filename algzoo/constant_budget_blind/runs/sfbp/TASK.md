You are an interpretability investigator. Your working directory is /tmp/claude-0/-home-user-clean-neurons/aa1e53e7-191b-5474-9620-48e6f2b9b20d/scratchpad/t24/sfbp . Work ONLY inside it.

THE MODEL. A trained one-layer ReLU RNN (no biases) reads a sequence of 10 real numbers x0..x9, each drawn
i.i.d. from N(0,1), and outputs the position (0..9) of the SECOND-largest number. It has 32 hidden neurons.
The final hidden state after reading x9 is h_10 (a vector of 32 non-negative numbers); the output is a
fixed linear readout of h_10.

YOUR TASK. Name what each of the 32 neurons of h_10 does, and write a formula for each.
For every neuron i = 0..32-1, produce:
1. a category: A = one main job (with related offshoots), B = two related jobs, C = several unrelated jobs,
   D = not understood;
2. a label: a plain-language description of the neuron's job, at most 40 words, self-contained (define any
   symbol you use), with positions written 0-based as x0..x9. The label must NOT mention the neuron's index,
   other neurons, how many neurons the model has, or the model's size; it will be read on its own;
3. a formula f(i, x) that predicts h_10[i] from the input x, implementing your label. It may use any
   interpretable features of x (values, ranks, sorted values, positions of the max/second, running maxima,
   comparisons, indicators, ReLU, ...) and AT MOST 5 FITTED REAL CONSTANTS per neuron. Discrete choices you
   searched for (position sets, which feature to use, thresholds you tuned) are allowed but must be declared.

YOUR ONLY ACCESS TO THE MODEL is the file /tmp/claude-0/-home-user-clean-neurons/aa1e53e7-191b-5474-9620-48e6f2b9b20d/scratchpad/t24/sfbp/probe.py:
  import probe; probe.hidden(x) -> (N, 32) final hidden states; probe.predict(x) -> (N,) predicted positions.
Run python as:  cd /tmp/claude-0/-home-user-clean-neurons/aa1e53e7-191b-5474-9620-48e6f2b9b20d/scratchpad/t24/sfbp && source /home/user/clean-neurons/.venv/bin/activate && ulimit -v 2500000 &&
  OMP_NUM_THREADS=1 python your_script.py
Do not open probe.py's internals, do not access weights, do not import alg_zoo or torch in your own code, and
do not run the recurrence yourself. You may generate any inputs you like (Gaussian or crafted), with any seed
EXCEPT 20261016. Keep each run light (<2 GB memory, at most a few hundred thousand sequences per call).

BLINDING RULES. Do NOT read, list or search any file outside /tmp/claude-0/-home-user-clean-neurons/aa1e53e7-191b-5474-9620-48e6f2b9b20d/scratchpad/t24/sfbp (in particular nothing under
/home/user/clean-neurons or /home/user/alignment-research-center, and no other directory under the parent of
/tmp/claude-0/-home-user-clean-neurons/aa1e53e7-191b-5474-9620-48e6f2b9b20d/scratchpad/t24/sfbp). Do not use the web. Do not start sub-agents.

EFFORT BUDGET. You have at most 80 tool calls in total (every Bash/Read/Write/Edit call counts).
Plan for it: spend most on probing and fitting, and reserve the last ~8 calls to write and check the
deliverables. Stop when the budget is used, even if some neurons remain unclear (label them D, with your best
formula).

DELIVERABLES (in /tmp/claude-0/-home-user-clean-neurons/aa1e53e7-191b-5474-9620-48e6f2b9b20d/scratchpad/t24/sfbp):
- labels.py defining
    LABELS = {i: ("A"|"B"|"C"|"D", "label text")} for every i in range(32)
    CONSTANTS = {i: number_of_fitted_real_constants}   (each <= 5)
    STRUCT = {i: "short note of discrete choices you searched, or ''"}
    def f(i, x): x is an (N, 10) float array; return a length-N float array predicting h_10[i].
  labels.py may import only numpy (and math). It must not contain the strings "torch", "alg_zoo",
  "weight_hh", "weight_ih", "state_dict" or ".pth" anywhere, and must not import probe.
  Hard-code the fitted constants in the file.
- notes.md: your method in a few paragraphs, the per-neuron R^2 you measured on your own fresh inputs, any
  overall picture of how the network computes the answer, and a list of every file you opened.
Before finishing, check that `python -c "import numpy as np, labels; x=np.random.default_rng(1).standard_normal((1000,10)); print([labels.f(i,x).shape for i in range(32)])"` runs.

Your final message: one paragraph summary, the per-neuron categories, and your number of tool calls used.
