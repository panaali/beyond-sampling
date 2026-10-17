# Frozen native sorting/search closure control

The installed venv reports NumPy2.5.4 git revisiona3d0ef1df045b858e68115001fdabc930bb3b68c.
Source downloads at that exact revision are saved with hashes and initial404s.
An accidental system-Python metadata probe (2.3.5) is preserved as installed.json;
installed_venv.json corrects it. All Hopf controls ran the mandated venv.

Source item_selection.c lexsort sorts each key stably in increasing key order,
carrying indices from the previous key. npysort_methods.cpp assigns float64
and int64 stable argsort to atimsort; only bool/8/16-bit integer use radix.
The exact numeric part of timsort.hpp can be compiled with counting Tag::less
and otherwise unchanged branches. This is a shadow source comparison census,
not a replacement estimator and not proof of compiler/binary equivalence.
Actual NumPy output remains authoritative and must equal the shadow permutation.

Freeze before execution: all permutations of lengths0..6; fixed duplicate,
reversed, increasing and modular-arithmetic keys at lengths15,16,31,32,63,64,
65,127,128,257,1024; float signed-zero/NaN/infinity order controls; paired
lexsort angle/id controls with duplicate keys. The shape and allocation cap
are fixed. Never use model outcomes to choose comparator/bounds. One thread,
2GB, <=20CPU. All failures retained. Independent source gate precedes controls.

Numerical comparison expression for float less is exactly
`a<b || (b!=b && a==a)`, so at most3 scalar comparisons per call; integer
less uses1. Counters reconstruct short-circuit scalar comparison counts.
Boolean control, integer index/merge bookkeeping, moves and allocation remain
excluded, not relabeled scalar numerical work. Default4-label argsort uses
quicksort/SIMD and is NOT priced by timsort; it remains a separate blocker.

The no-sorter binary-search source processes all keys with common interval
length, first ceil-halving and final comparison. For N>0 each key executes
max(ceil(log2 N),1)+1 comparator calls; N0 has zero. Right-side comparator
is !less(key,pivot). A safe numerical subset bound is3 comparisons per call,
with explicit Boolean inversion separately1. Source/native analytic right
insertion parity must pass before contracting this bound into saved ledgers.
