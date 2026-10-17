"""Retrospective, formula-only completion of the frozen M_32,10 draft.

[inferred] Categories describe the apparent jobs of the existing formulas; they
are not independent evidence that the true neurons have those jobs. These labels
were assigned without viewing held-out scores, activations, or model weights.
The original investigator left LABELS empty, so this is a retrospective
completion, not that investigator's preregistered qualitative judgment.

Rubric from PREREGISTRATION.md: A = one main job with related offshoots;
B = two related jobs; C = several unrelated jobs; D = not understood.
Here D means the formula does not establish a coherent one- or two-job account;
it does not assert that its ranking-related terms are unrelated (category C).
No category is determined by predictive performance. A/B labels still require
the separate held-out R-squared criterion to count as legible.

All positions below are zero-based; "prefix" means x0..x8. "Gap" means its
largest minus second-largest value. The formulas were already selected and
fitted. Position subsets, selected features, thresholds, and functional forms
carry selection complexity beyond the number of fitted scalar constants.
Neither their search cost nor their description length is known here.

Only LABELS is completed. f is reexported unchanged from labels_32_10.py.
"""

import importlib.util
from pathlib import Path


_source = Path(__file__).resolve().parents[1] / "labels_32_10.py"
_spec = importlib.util.spec_from_file_location("_e5_frozen_original", _source)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load frozen formulas from {_source}")
_original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original)
f = _original.f


LABELS = {
    0: ("B", "Combines a deficit of the last input below the prefix runner-up with gap coding for maximum positions {1,2,5,6} and {0,2,3,4,5,6}, suppressed at position 8."),
    1: ("A", "Encodes the prefix maximum-to-runner-up gap, suppressed when the maximum is at position 8."),
    2: ("B", "Combines position 3's top-two membership and value relative to the third-largest prefix value with a maximum-value code for positions {0,3}."),
    3: ("D", "An unresolved mixture of gap coding at maximum positions {0,1,2,3,4}, runner-up position membership, position 8 top-two suppression, and maximum magnitude at {0,3,8}."),
    4: ("D", "An unresolved mixture of runner-up magnitude and position groups, maximum magnitude at {1,2,4}, and second-to-third gap gated by runner-up position."),
    5: ("A", "Encodes the prefix gap with increased gain for maximum positions {0,3,4,8} and an additional increase at {1,8}."),
    6: ("B", "Combines a deficit of the last input below the prefix runner-up with gap coding for maximum positions {0,2,5,7} and {0,2}."),
    7: ("A", "Detects the last input exceeding the prefix maximum, with a small fitted offset."),
    8: ("D", "An unresolved mixture of maximum magnitude at positions {0,1,2,5}, position 5's excess over competitors' runner-up, the first-five maximum, and the third-largest prefix value."),
    9: ("B", "Combines low prefix runner-up and fourth-largest values with position 8's excess over the runner-up among its competitors."),
    10: ("B", "When the prefix maximum is at position 4 or 5, combines its magnitude with a penalty increasing with the runner-up's position index."),
    11: ("D", "An unresolved mixture of maximum magnitude at {0,4,6}, runner-up magnitude gated by a different maximum-position set, and position 4's lead over all competitors."),
    12: ("A", "Detects the last input falling below the prefix runner-up, with a small fitted offset."),
    13: ("B", "Combines a deficit of the last input below the prefix runner-up with gap coding for maximum positions {1,2,4,7} and {0,2,4,6,7}, suppressed at position 8."),
    14: ("D", "An unresolved mixture dominated by position 8's deficit below competitors' runner-up, with position 0 excess, position 4 deficit, and position-gated prefix gap terms."),
    15: ("B", "Combines position 1's top-two strength with maximum magnitude additionally gated by runner-up positions {0,1}."),
    16: ("A", "Detects a weighted excess of the last input over the prefix runner-up."),
    17: ("A", "Encodes positive prefix maximum magnitude with a runner-up contribution and a small gain reduction at maximum positions {0,1,3,4,6,7}."),
    18: ("B", "Combines a deficit of the last input below the prefix runner-up with gap coding for maximum positions {1,4,6,8} and {0,1}."),
    19: ("B", "Combines position 6's top-two membership and associated value with an extra maximum-magnitude contribution at positions {0,2}."),
    20: ("D", "An unresolved mixture of position 0's deficit and top-two suppression, maximum magnitude at {1,8}, and a second-to-third gap gated by runner-up positions {1,2,4,6,8}."),
    21: ("B", "Combines the last input's weighted excess over the prefix runner-up with gap suppression that is stronger for odd maximum positions."),
    22: ("A", "Encodes position 7's competitiveness for the prefix top two through its value, excess over competitors' runner-up, and maximum-position bonus."),
    23: ("B", "Combines the last input's weighted excess over the prefix runner-up with stronger gap suppression at maximum positions {2,6,7}, and is suppressed at position 8."),
    24: ("D", "An unresolved mixture of position 5's deficit and top-two suppression, the first-eight runner-up value, and position 6's deficit below its strongest competitor."),
    25: ("D", "An unresolved mixture of position 8's top-two suppression and deficit below competitors, with runner-up magnitude weighted by two overlapping runner-up-position sets."),
    26: ("B", "Combines position 2's top-two strength with a distributed maximum-position code using magnitude at {0,2,4} and gap at {0,2,7}."),
    27: ("A", "Encodes prefix maximum magnitude and gap only when its position is 2 or 6, above a small baseline."),
    28: ("D", "An unresolved mixture of maximum magnitude gated by overlapping position sets, another maximum-position indicator, and position 8's deficit below competitors' runner-up."),
    29: ("B", "Combines a deficit of the last input below the prefix runner-up with gap coding for maximum positions {0,3,6,7} and an extra gain at 3, suppressed at position 8."),
    30: ("D", "An unresolved mixture of position 4's deficit and top-two suppression, runner-up magnitude, and maximum magnitude gated by positions {0,3}."),
    31: ("B", "Combines prefix runner-up magnitude with a gap contribution for maximum positions {0,1,3,4} and a small maximum-magnitude contribution."),
}
