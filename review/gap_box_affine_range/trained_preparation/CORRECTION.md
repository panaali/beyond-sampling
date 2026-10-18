# Preserved written-review correction

Initial source8a9846b3/shared05c6efc7 stated the feasible second-coordinate interval as[3/4,1]. Independent written reviewer recover_fundamentals identified that this incorrectly used the other coordinate upper endpoint in the lower-feasibility condition. For the full selected box intersected with the labeled gap constraints, the correct interval is[1/2,1]. The clipped intervals change formula at t=3/4; this is an interior kink, not a feasibility endpoint.

The intended domain remains the full selected box intersected with the fixed ordered-pair gap1/4. No extra t>=3/4 restriction is imposed. All four frozen bin tuples, indices, gaps and source bytes remain unchanged. This is a correction to the written description before implementation, trained payload decoding or candidate execution. Preserve the original commit and reviewer finding; no successful or failed numerical attempt exists in this preparation phase.
