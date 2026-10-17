# Cold dimension provenance: candidate static closure

The original saved `lastN` observation is insufficient and remains preserved in
cold_dimension_gap.json. This is a separate source/installed-code argument, not
an inference from that field or a new numerical replay.

The pinned run resets before load(False), executes only that loader and halves(4),
then records cold and resets. Loader AST retains definitions/classes/assignments,
so original module imports/main code do not execute. Top-level numerical work is
four leggauss rules (12,48,24,96). halves(4) forms cones with four-coordinate rays;
all matrix_rank calls select rows of matrices with four columns. Model-dependent
rank calls have the same four-column invariant, also supported by the saved
GESDD shape census. Inverse and vector norm calls do not initiate SVD.

Installed leggauss calls eigvalsh on a real symmetric companion. Installed
eigvalsh defaults UPLO=L and calls eigvalsh_lo. The pinned native eigvalshlo
forwards L to the constant-propagated eigh wrapper at0x8270. That wrapper sets
JOBZ=N: pointer rdi is rsp+0x178 at0x8329, then rsp decreases8 and the same byte
is written0x4e at new rsp+0x180 (0x8372). Both query/execute call DSYEVD with this
structure. DSYEVD tests JOBZ against V (rodata0x13bf6f9, objdump byte0x56), storing
WANTZ in r12. After DSYTRD0xd8079a the test0xd807a3 selects DSTERF0xd807c4 when
WANTZ=false. The vector path instead uses DSTEDC; it is not selected here.

cold_helper_callgraph.json follows direct calls below DSTERF and DSYTRD's
DSYTD2/DLATRD reductions. None reaches DQDS. BLAS operations and ordinary error/
runtime handlers are explicit boundaries; prior installed norm/reflector helper
pins are consistent. This argument assumes ordinary ABI semantics/fixed binding
for these routines, not arbitrary symbol interposition or malicious callbacks.
The full-library direct reverse scan additionally identifies DLARRE as another
Q2 caller, not merely Q1; that caller is absent from the selected DSYEVD path.
Therefore no inference that all eigensolvers avoid DQDS is made.

For remaining matrix_rank SVD calls, pinned GESDD JOBZ=N reduces to bidiagonal
order min(m,n)<=4, then DBDSDC/DLASDQ/DBDSQR/Q1/Q2 preserve or reduce that order
(as reviewed in rank_cost_review). The cold 192 Q2 calls consequently belong to
small rank geometry under this static argument, despite large leggauss inputs.

Status: candidate installed/source provenance, awaiting independent review.
No global Q1/Q2 numeric contraction is accepted yet. Historical binary/binding
provenance remains conditional, as for the other finite native prices.
