# T66 native four-input lifecycle: source review

Compared the owned D4 implementation against the previously reviewed D3
lifecycle, including wrapper and frozen check script. D=4 consistently sets
hidden-map strides,16-entry readout maps, four-coordinate rays and4x4
simplices, rank-test columns, serialization shapes, and recursion termination.
Counts have10 entries: four stage counts, two outcome-piece counts, two
simplex counts, rank events and clipping events. Gate constraint IDs1..48,
readout49..52, and rank53..56 do not overlap and fit the128-bit capacity.
Future map columns remain zero by induction from initialization; each step
replaces past columns and writes only its new input column.

LAPACK receives selected rays as an m-by4 column-major matrix with leading
dimension max(1,m), no singular vectors, and the inherited sufficient workspace.
The same LP64/32-bit integer ABI assumption applies. NumPy's actual input-gate
ordering is passed to native code, preserving the earlier tied-sort repair.
The wrapper copies every returned array before freeing the native handle.

For fixed common gates, the real-arithmetic componentwise map discrepancy
bound2*((1+gamma)^3-1)*M follows from three recurrence products, where M is
the nonnegative absolute-weight propagated envelope. The script's computed
M and extra(1+gamma)^3 factor are ordinary floating bound arithmetic, not
directed certification. If gamma were itself a proven relative loss bound
for computed M, a strict correction would be(1-gamma)^-3; spare conservatism
from using machine epsilon rather than unit roundoff would need an explicit
proof. The owner retains this qualification rather than changing a frozen
numeric threshold after observing results.

Reported finite controls and six H2/H4 cases pass ordered topology with max
ray difference1.101e-13 and map-bound ratio .0265. This memo independently
reviews the dimensional/source argument, not a second full workload. The
initial zero-measured-CPU reporting division failure remains recorded; null
CPU ratio plus separately measured wall time is an appropriate reporting
repair. Speed applies to this geometry implementation only. No mass-rule,
T5+, floating-topology certificate or complete sampling comparison follows.
