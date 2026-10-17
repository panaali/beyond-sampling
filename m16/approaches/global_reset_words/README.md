# T50: exact global reset words for mixed-sign H2

## Question
Does the public H2T10 model have open sets of four-input words that reset every initial nonnegative hidden state, improving on the known five-positive-input sufficient event? Claim published and rechecked bb4cff09. The prior T50 owner confirms this fixed-word global-image classification was not attempted; T49 separately owns the same-sign two-input reset wedge.

## Method
Represent arbitrary initial hidden state through homogeneous coordinates y=(s,h01,h02)>=0, sum(y)=1. Physical finite states have s>0 and hidden h0=(h01/s,h02/s); the word becomes s*x. Start affine hidden map B=[[0,1,0],[0,0,1]]. Propagate exact Fraction gate polygons and affine maps in this compact triangle. Retain s=0 recession vertices; discard only exact zero-area cells. Continuity recovers their boundary values from adjacent full cells. Global reset holds iff final B*v=0 for every vertex of every full cell. This certifies unbounded initial states, not just a large sampled box.

For robust reset, retain last preactivation L on each cell. Each finite vertex gives −L_i(v)/s(v); recession vertices must satisfy L_i(v)<=0. Their minimum eta_i certifies L_i(y)<=−eta_i*s(y) globally. The last preactivation changes by at most s*epsilon*C_i under any word perturbation of infinity norm epsilon, where C=sum_(r=0..length−1)|Wh|^r |Wi|. Strict positive eta gives an open reset box epsilon<min eta_i/C_i without assuming gates stay fixed. Failure of this sufficient margin certificate is not proof that no open reset set exists.

## Status
Protocol frozen before implementation. Exact floating checkpoint coefficients are treated as binary rational numbers. Bounded numerical cycles <=110CPU, one thread, <=2GB, shared compute lock; <=20CPU controls may use review lock. No shared edits, full accuracy estimator or Monte Carlo.

## Frozen controls and experiments
1. Known toy maps: Wh=0 with mixed-sign Wi resets iff final input is0; zero Wi and Wh=0 always resets; positive diagonal Wh preserves an unbounded recession direction for every finite word; inhibitory first-row toy Wi=(1,1), Wh=[[0,−1],[1,−1]] resets on two negative inputs. Include empty-word identity.
2. Verify exact cell coverage by rational signed areas summing to the initial triangle area; check direct exact recurrent execution at each cell centroid and finite vertices. First-index/readout is irrelevant to hidden reset; zero gates have one inactive owner.
3. Public H2T10 seed0: verify known two-positive-axis and three-zero-input facts using exact weight inequalities. Known five-positive word(1,1,1,1,1/20) must pass its independent sufficient theorem. Test the32 four-input words formed by all16 sign patterns times magnitude patterns(1,1,1,1) and(1,1,3,1). Do not substitute other words based on outcomes. Record exact witnesses, margin radii, cell/vertex/recession counts, rational arithmetic work and CPU. Preserve all failures and incomplete cases under a100CPU guard.
4. A word classified nonreset must supply a finite rational initial-state witness, including conversion from a nonzero recession vertex via a neighboring finite vertex. Evaluate that witness independently through the exact recurrence. A reset word with positive radius will be checked at all16 corners of a half-radius word box; this is validation of a proof, not its basis.

Alternatives before results: four-step open resets exist and a short certificate reduces the effective history; or the fixed panel misses them/none exist in this panel, while the known five-step witness still passes. An implementation error can masquerade as reset if recession rays or zero-area/gate boundaries are mishandled; exact partition and witness controls discriminate that. No finite search proves global minimality.

## Results
Pending. Commands and artifacts will be added per cycle. Success here means a valid structural certificate, not an accuracy estimate.

## Compute accounting
Count exact rational dot/clip/matrix operations, produced cells/vertices, maximum rational bit lengths, direct validation recurrence steps and wall/CPU setup. Fraction operations are not ordinary FLOPs; no portable equal-compute win is implied. Include checkpoint load and failed-case work separately. No random model-input samples.

## Claims
[inferred] Projective full-cell vertex test certifies global fixed-word reset by homogeneity, linearity and continuity (independent T55 algebra review). [speculative] A robust four-input word could improve the earlier five-positive reset event. Author results require independent review before DRI confirmation.

## Alternatives not ruled out
Other magnitude patterns, longer words, exact parameterized input-word cones, and sharper robustness norms remain open. A reset changes hidden-state dependence but does not remove discarded-input rank dependence; no earlier independent negative-prefix rank law is reused.

## Failure modes
Exact rational cell count/bit growth; incomplete coverage; missing recession directions; insufficient strict margin despite a real open reset set; a narrow fixed-word panel missing useful cones. General model-state transport, target-rank augmentation and probability of overlapping word events are out of this initial phase.
