# Conditional source envelope for four finite float64 labels

Frozen source: NumPy revision a3d0ef... and its x86-simd-sort submodule
fa944efbea1a33426b3f8a9a21e23794ab8aa300. Scope: default ascending argsort,
contiguous per-row float64 length4, all input keys finite; no custom dtype,
NaN fallback, descending mode or arbitrary length. Actual native results
remain unchanged. This is a source-path envelope, not a hardware instruction
or complete runtime cost claim.

`quicksort.hpp` can dispatch to AVX512/AVX2 argsort or use scalar insertion
for length4. `x86_simd_argsort.dispatch.cpp` passes hasnan=true. The SIMD entry
first scans for NaNs, then tests std::is_sorted. For unsorted length4 it sets
max_iters=2*log2(4)>0 and enters the <=256 network case. The recursive
argsort_n_vec halves numVecs until one register; no inter-register merge then.
The eight-lane network performs exactly6 cmp_merge stages. Each stage executes
one min, one max and one equality per lane; shuffles/masked moves are memory/
control operations outside this numerical subset. Padding lanes count too.

An inclusive numerical subset bound per four-key row is:
-8 lane classifications in the masked NaN scan;
-3 adjacent numerical comparisons for is_sorted;
-6*8*3=144 lane min/max/equality operations for the network;
-1 log2 and1 integer multiplication setting max_iters.
Total157. The four-lane AVX2 network has3 stages and36 lane operations,
so the same157 bound covers it. Scalar fallback insertion has at most6
Tag::less calls, at most18 scalar comparisons, also below157. The AVX2
classification implementation and generic source/binary dispatch correspondence
must remain explicit qualifications unless independently established.

This bound omits index/mask construction, branch/loop bookkeeping, allocation,
gather/shuffle/move work, and native-library dispatch. It cannot be called a
full operation total. No value-error or statistical bound follows.

Frozen control before use: all24 permutations of four distinct finite keys,
all81 tuples over{-1,0,1}, and signed-zero layouts. Reconstruct the exact
source's padded eight-lane network, including index ties and early sorted
exit; compare original NumPy default argsort permutation. Preserve any failure,
never replace outputs or choose a bound from measured ratios. Check source
stage count and metadata hashes independently. Keep all nonfinite inputs
outside this bound, even though other source-shadow controls test their order.
