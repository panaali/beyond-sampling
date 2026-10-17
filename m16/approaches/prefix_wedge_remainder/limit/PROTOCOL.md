# Reset-limit value gate

Renewed claimb0db2209 precedes code. Parent first-phase proof/results independently38d50e9c. Only saved trainedweights/application are reused; no checkpoint reload, originalnetworkforward or MC.

The limit final preactivations are ai+bi*r, ai=Wh[i,1]*Wi1, bi=Wi[i], r=y/u andu>0. Freeze exact Fraction partition at every nonconstant gate root. Within each open gate interval, freeze every pairwise readout affine root, retaining persistent equalities. Classify each open subinterval with an exact rational representative and first-index argmax. Evaluate every finite boundary separately for tie ownership; boundaries have Gaussian measurezero, persistent ties do not. Keep intervals explicit rather than discarding tie cases.

CDFs: F10(c)=P(u>0,x>u,y<cu)=1/8+[-pi/4+atan(c)−atan(c/sqrt(2+c²))]/(4pi). F01 has +pi/4 and +atan. Negative-infinite endpoint is0; positive-infinite endpoints1/8 and3/8 respectively. Intersect correctrintervals withr<1 forrank10 andr>1 forrank01. Sum CDF endpoint differences and multiply1/2 for independentnegativeprefix. This is joint correctness mass on the sector of probability1/12.

Use reviewed rational atan/sqrt/pi functions. Outward-round EACH CDF interval to a160bit dyadic lattice with exact integer floor/ceiling before summing, to prevent unnecessary denominator growth. This preserves enclosure, never rounds using binaryfloating functions. Freeze four rounding controls: point1/4, point1/3, interval[-2/3,-1/3], interval[-1/3,2/3]. Exact containment and floor/ceiling distances checked. This is new small primitive source gate.

Nine frozen partition controls, targetindex2, each two hidden units/four outputs. A/B are affine intercept/slope; W rows ordered0..3:

1. Allzero A/B/W: mass0, persistentfirstindex0.
2. A=(1,0),B=(0,0), W=((0,0),(-1,0),(1,0),(0,0)): mass1/12.
3. SameA/B,W=((0,0),(1,0),(1,0),(-1,0)): earlierpersistenttie, mass0.
4. SameA/B,W=((0,0),(-1,0),(1,0),(1,0)): laterpersistenttie, mass1/12.
5. A=(0,0),B=(1,-1),W=((0,0),(0,1),(1,0),(-1,-1)): targetwinsr>0, mass5/96; at0winner0.
6. SameA/B,W=((0,0),(1,0),(0,1),(-1,-1)): targetwinsr<0, mass1/32.
7. A=(1,0),B=(0,1),W=((0,0),(1,0),(0,1),(-1,0)): targetwinsr>1, mass1/24, exactr1tieearlierindex1.
8. SameA/W,B=(0,-1): targetwinsr<−1, mass1/48, exact−1tieearlierindex1.
9. A=(-1,-1),B=(1,-1),W=((0,0),(-1,-1),(1,1),(-2,-2)): targetwins|r|>1, disjointintervals mass1/16; zero-support endpoints/interiorwinner0.

For each control, preserve all openintervals, gatewords, pairroots, exact boundary winners and rational probability enclosure; known exact mass must be enclosed. At least one positive andnegative rational point per relevantinterval, exact gate/marginroots already cover continuum. No random controls or posthocfixture changes. Independent source/control audit before applying to saved trainedweights.

Then one unchanged saved trained case, pins for parentapplication, originalcontrols, reviewedprimitive and newcontrols. Report limit jointcorrectness enclosure, sector-only originaljoint enclosure enlarged by parenteventupper, and conditional sectoraccuracy only if explicitly divided by1/12. Keep remainderupper as an upper onpossible disagreement, notobservederror. Exact representative operations are algebraic validation of the reduced formula, notoriginalnetworkforward samples. Record named arithmetic/bitlengths/CPU and no sampling-advantage claim. <=20CPU review,2GB,one thread.
