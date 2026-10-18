You are a judge. Below are short labels, each written by an investigator to describe what ONE hidden neuron of a
small recurrent network does. The network reads 10 real numbers x0..x9 (i.i.d. Gaussian) and outputs the
position of the second-largest number. Judge each label ON ITS TEXT ALONE. Do not use any tools and do not
read any files; answer directly.

For each item give:
1. CAT, the legibility category the text supports:
   A = one main job (with related offshoots); B = two related jobs; C = several unrelated jobs; D = not
   understood (the text does not give a coherent account of the neuron's job).
2. ROLE, the single PRIMARY role from this frozen list, and optionally ROLE2, a secondary role (or "-"):
   MAX        magnitude of the largest value (of the whole sequence or of a prefix)
   SECOND     magnitude of the second-largest value (or third-largest)
   GAP        a difference between top-ranked values (largest minus second, second minus third), i.e. a margin
   POS_MAX    which position holds the largest value (a position code, possibly scaled by a magnitude or gap)
   POS_SECOND which position holds the second-largest value, or whether a specific position is among the top two
   LAST       a comparison involving the last input x9 (x9 above/below the prefix max or runner-up, x9 new max)
   RAW        raw input value(s) at fixed positions, without ranking
   OTHER      anything else, or a mixture with no dominant role
   Choose the role that the label says carries most of the neuron's activity. If a label says "the gap,
   with a gain depending on where the max is", the primary role is the one the text emphasises; if both
   are equally emphasised prefer the first mentioned and put the other in ROLE2.

Answer with exactly one line per item, in item order, in the format
ITEM_ID | CAT | ROLE | ROLE2
and nothing else.

ITEMS:
__ITEMS__
