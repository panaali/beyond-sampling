# T136: independent raw-leaf triangulation review

## Question
Does source pulling triangulation lose or duplicate volume within the raw terminal cones of the four fixed H8T4 seed0 parent halves0,1,20,21? Claim3e02ca4 was pushed before implementation; this is a distinct T131 review coordinated with recover_910.

## Method
Pin T131 estT_snapshot.py SHAafa94627ff5b860a4b922dc340e90d0d6718c770d74e326dfbf667a2bb0bbc3a, m43.py dependency and public checkpoint. Import source locally without edits. Disable Gaussian mass and accuracy evaluation: replace _cell_error with a raw R/N/T capture callback, and orthant with a simplex recorder returning0. Invoke exact _finish for only the four named halves. Source recurrence, clipping, incidence and pulling code stay unchanged. Save full raw leaf vertices/facets/incidences before triangulation, not just simplices.

For each parent, anchor a=sum(parent inward normals) is strictly positive on its rays. Intersect each leaf ray with a·x=1, project into an independently constructed orthonormal3D basis of that affine plane, and compute ConvexHull volume from ALL raw vertices. Compare with source pulling tetrahedral volumes. Independently check the4D determinant relation volume3=||a||*abs(det(rays))/(6*product(a·ray)). Check every raw leaf ray against all leaf and parent inequalities, and record symbolic incidence residuals. Sum leaf hull volumes versus parent hull volume, retaining all signed and absolute differences.

Freeze numerical controls first: standard positive orthant (section volume1/3 for anchorones), unit cube cone (volume1 with anchore4), thin boxes with widths2^-10,2^-20,2^-30 and2^-45, and a fixed orthogonal Hadamard rotation. For every hull also compute an independently conditioned version: center all projected vertices, full3D SVD coordinate transform (no rank truncation), ConvexHull in whitened coordinates, multiply volume by product singular values. Preserve raw QHull failures, singular values, rank/conditioning and both results. No random QHull joggling. A flat control is explicitly measure-zero and may be rejected by QHull; it is not silently treated as valid full-dimensional geometry.

Fixed trained diagnostic tolerances: report every raw/conditioned per-leaf absolute and relative discrepancy; use <=1e-10 parent volume per leaf and <=1e-9 parent volume summed absolute difference as finite compatibility screens, not certified error bounds. If controls/frozen leaves fail, retain them and report limitations before any new rescue. No new parent selection, model/noise sweep, full-geometry enumeration or Gaussian mass integration.

Alternatives: (1) source pulling omits a face or duplicates simplices: independent full-vertex hull volume can expose a discrepancy; (2) clipping already omitted extreme rays: both representations can agree, so this check cannot rule it out alone; (3) signed quadrature bias causes the mass mismatch despite good affine volumes: no Gaussian mass is evaluated here. Equal volumes alone do not prove absence of overlapping and compensating missing regions or certify all floating topology.

## Status
Protocol frozen; analytic controls completed, raw-leaf capture next.

## Results
Eleven frozen analytic controls completed. Orthant, unit boxes and unrotated widths down to2^-45 agree to about1.6e-15 relative. Hadamard-rotated width2^-45 exposes floating input/projection precision: source error .00108507, raw hull .00121088, conditioned hull .00206163 relative to ideal volume (absolute errors below6e-17). This failure is preserved in controls.json; SVD cannot recover bits already lost before conditioning. Flat control is rejected as expected. The trained fixed-parent screen remains a parent-relative finite diagnostic, not an accuracy guarantee for extremely thin individual leaves. Command: one-thread Python review.py controls (0.206CPU seconds including imports).

## Compute accounting
One thread,2GB, review lock<=20CPU per bounded process, checkpoints after each parent. Charge source setup, clipping, triangulation, all hulls/SVDs and persistence in measured CPU; source counter is a legacy geometry proxy, not full FLOPs. No random network inputs or trained-model forward sampling.

## Claims
[inferred] On a strictly positive affine section, cone triangulation volume consistency can be checked without Gaussian integration.
[speculative] The fixed leaves may reveal whether pulling triangulation contributes to T131's mass discrepancy.

## Alternatives not ruled out
Uncaptured parents, clipping omissions, offsetting overlaps/gaps, floating incidence errors below this screen and Gaussian integration bias. A passing finite panel is not a universal topology certificate.

## Failure modes
Very thin projected hulls can exceed QHull precision; SVD conditioning can amplify input roundoff. Source rays and incidences remain floating inputs. Equal aggregate volume can hide local defects, and shared source capture is not independent neural geometry enumeration.
