# Neuron labels for the 32-neuron second-largest RNN (h_10)

## Method
All access went through `probe.hidden` / `probe.predict` on inputs I generated myself (Gaussian seeds 1, 5, 7, 11, 12, 21, 22, 31, 41, 777, plus some crafted inputs: single impulses and all-negative sequences). The model's accuracy on seed-1 data is 0.991.
1. Screening (s1.py): for each neuron, the active fraction, correlation with each x_j and with the sorted values, and the mean activation grouped by the position of the max and of the second-largest.
2. Feature library (lib.py / featlib.py): inputs, sorted values S0..S2, prefix and suffix maxima, the prefix second-largest, the max and second-largest of the other inputs, and ReLU "margins": isMaxK (lead of xK over all others), inTop2_K (lead of xK over the second-largest of the others), belowK (how far xK is below the max of the others), recK (xK beats all earlier inputs), plus group maxima over position sets.
3. Each formula is ReLU(sum_k c_k * feature_k) **without an intercept**. The model has no biases, so h(c x) = c h(x) for c > 0, and every feature is chosen to be degree-1 homogeneous. Fits use alternating least squares on the active set (rfit.py). Features were picked by greedy forward selection (s4, s8, s11, s15), by local search over position partitions (s5, s10, s16, s17), and by targeted hypothesis tests (s7, s9, s13, s18-s20, s22, s23). For each neuron I kept the simplest candidate within 0.01 validation R^2 of the best (s14, s21). The discrete choices (feature names and position sets) are recorded in STRUCT.
4. Validation: the R^2 below is measured on 100k fresh Gaussian inputs (seed 777, s24.py) that were not used for fitting.

## Per-neuron R^2 (fresh data, seed 777)
| n | cat | #const | R^2 | | n | cat | #const | R^2 |
|---|---|---|---|---|---|---|---|---|
|0|B|5|0.736| |16|D|2|0.069 (almost always 0; var 0.003)|
|1|A|4|0.902| |17|A|4|0.985|
|2|A|5|0.914| |18|A|5|0.971|
|3|A|4|0.975| |19|A|4|0.980|
|4|B|5|0.940| |20|A|3|0.969|
|5|D|5|0.845| |21|A|4|0.910|
|6|A|4|0.984| |22|A|5|0.979|
|7|A|2|0.997| |23|A|4|0.862|
|8|A|2|0.994| |24|B|5|0.926|
|9|A|3|0.994| |25|D|2|0.695 (almost always 0; var 0.004)|
|10|D|5|0.435| |26|B|5|0.830|
|11|A|4|0.914| |27|A|2|0.995|
|12|B|5|0.863| |28|A|5|0.987|
|13|A|3|0.992| |29|B|5|0.687|
|14|A|4|0.974| |30|A|5|0.979|
|15|A|5|0.863| |31|B|5|0.703|

## Overall picture
- **"How far x9 is below the max" family** (3, 7, 9, 17, 18, 20, 22, 28, 30; and 14 for x8, 24 for x7): ReLU(a*(max of the others - x_k)), with a position-dependent correction. When the max sits in a neuron-specific set of positions, the max's lead over the runner-up is subtracted, so the neuron then measures (runner-up - x9) instead of (max - x9). Different neurons use different position sets. Taken together, they seem to tell the readout whether x9 is below the second-largest value and where the max sits.
- **"New maximum" detectors** (8: x9 beats x0..x8; 27: x8 beats x0..x7) and **"top-two" detectors** (13: x9 is in the top two).
- **Rank detectors for single positions** (21: x1, 1: x2, 11: x3, 2: x4, 23: x6, 26: x7, 29: x0). The mean activation falls steeply with x_k's rank (rank 0 about 9-14, rank 1 about 4, rank 2 about 1.6). They are modelled as a*isMax_k + b*inTop2_k plus a smaller term, often in max(x0, x_k). x0 shows up oddly often: inTop2_0 / isMax0 terms appear in 0, 15, 24, 26 and 29.
- **Max-position / max-level codes** (19: about 10*max + 5*second-largest; 15: early maximum; 6: max in x4..x8; 4: max in middle positions; 12: max in {2,3,4,6,7}, zero when the max is x5).
- **Weak or rare neurons**: 5, 10 and 31 are small and only partly understood. 16 and 25 are essentially silent on Gaussian inputs: they fire only when every input is negative.

## Caveats
Several formulas are empirical: 4-5 greedily chosen features whose combined meaning I describe in the label. The fits for 0, 10, 29 and 31 are weak (R^2 < 0.75). Group sets and feature choices were searched on the same Gaussian distribution, but validation used held-out seeds.

## Files opened / created
Opened (read): TASK.md only (probe.py was imported, never read). Created in this directory: s1.py ... s24.py, feats.py, feats2.py, lib.py, featlib.py, rfit.py, gen.py, d1.npz, s4.out, s5.out, s8.out, s10.out, s11.out, s10.json, s11.json, s17.json, chosen.json, final_spec.json, labels.py, notes.md. I also read my own background-task output file (s3 output) in the session tasks folder.
