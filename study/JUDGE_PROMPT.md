You are helping with a neural-network interpretability study. Read ONLY the file {PACKET}. Do not open any
other file in the repository (in particular nothing under study/keys, study/explain*, or data/), and do not
run code.

The file has several items (IDs like J042). Each item gives an explanation of what one unit of a language
model responds to, followed by 12 numbered text snippets. Each snippet has one token marked [[like this]].
Using only the explanation, predict how strongly the unit fires on the marked token:
3 = strong (near its maximum), 2 = moderate, 1 = weak, 0 = not at all.
Judge each snippet on its own. Some explanations may not fit any of their snippets, and the mix of
strong and weak snippets varies by item.

Write a JSON list [{"jid": "J042", "ratings": [r1, ..., r12]}, ...] (integers 0-3, in snippet order)
covering every item in the file, to {OUT}, and nothing else. Reply with just "done N".
