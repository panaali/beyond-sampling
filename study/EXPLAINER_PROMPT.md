You are helping with a neural-network interpretability study. Read ONLY the file {PACKET}. Do not open any
other file in the repository (in particular nothing under study/keys, data/, or other batches), and do not
run code.

The file has several units (IDs like P017). Each unit is one direction in a language model's MLP hidden
layer. For each one you get text snippets where it fires. The token it fires on is marked [[like this]],
and the number in parentheses is the activation as a % of that unit's maximum. You see top examples (one
per document), mid-range (20-50% of max), and weak (5-20%) examples.

For every unit, work out what it responds to and write:
- "explanation": one or two sentences precise enough that someone who never saw these examples could predict,
  for a new snippet, whether the marked token makes the unit fire strongly, moderately, or not at all.
  Say what matters (the token itself, the preceding context, or both).
- "category", exactly one of:
  "A" = one main job (small offshoots allowed if clearly related to it),
  "B" = two related jobs,
  "C" = several unrelated jobs / a mixture,
  "D" = you can't tell what it does.
  Be honest. Don't force a single story onto examples that don't share one, and don't split a coherent
  concept into pieces.
- "confidence": 1-5.

Write a JSON list [{"id": "P017", "explanation": "...", "category": "A", "confidence": 4}, ...] covering
every unit in the file, to {OUT}, and nothing else. Reply with just "done N" where N is the number of units.
