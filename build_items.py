"""Build blinded explainer packets and (after explanations exist) judge packets.

usage: python build_items.py explain   -> study/explain/batch_XX.md, study/keys/items.json
       python build_items.py judge     -> study/judge/batch_XX.md,   study/keys/judge.json
"""
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer

A = "data/acts"
S = "study"
SKIP, BEFORE, AFTER = 4, 24, 4
N_EXPLAIN_BATCHES, N_JUDGE_BATCHES, N_CONTROL = 10, 10, 30

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
tokens = np.load(f"{A}/tokens.npy")
doc_of = np.load(f"{A}/doc_of.npy")
neuron_idx = np.load(f"{A}/neuron_idx.npy")
shape = tuple(json.load(open(f"{A}/meta.json"))["shape"])
n_done = json.load(open(f"{A}/progress.json"))["done"]
acts = np.memmap(f"{A}/acts.f16", dtype=np.float16, mode="r", shape=shape)
N_LAYERS, N_NEU = shape[2], neuron_idx.shape[1]
piece = {}


def tstr(t):
    if t not in piece:
        piece[t] = tok.decode([int(t)]).replace("\n", "⏎")
    return piece[t]


def snippet(w, p):
    ids = tokens[w]
    left = "".join(tstr(t) for t in ids[max(0, p - BEFORE):p])
    right = "".join(tstr(t) for t in ids[p + 1:p + 1 + AFTER])
    return f"{left}[[{tstr(ids[p])}]]{right}"


def pole_values(layer, unit, sign):
    return sign * acts[:n_done, SKIP:, layer, unit].astype(np.float32)


def per_doc_top(v, k):
    """Top-k (window, pos) by value with at most one example per document."""
    flat = v.reshape(-1)
    order = np.argsort(-flat)
    out, seen = [], set()
    for f in order:
        w, p = divmod(int(f), v.shape[1])
        if doc_of[w] in seen:
            continue
        seen.add(doc_of[w]); out.append((w, p + SKIP, float(flat[f])))
        if len(out) == k:
            break
    return out


def band(v, lo, hi, k, rng, exclude_docs):
    mx = v.max()
    ws, ps = np.nonzero((v >= lo * mx) & (v < hi * mx))
    perm = rng.permutation(len(ws))
    out, seen = [], set(exclude_docs)
    for i in perm:
        w, p = int(ws[i]), int(ps[i])
        if doc_of[w] in seen:
            continue
        seen.add(doc_of[w]); out.append((w, p + SKIP, float(v[w, p])))
        if len(out) == k:
            break
    return out


def sample_items():
    rng = np.random.default_rng(1)
    items = []
    for layer in range(N_LAYERS):
        for j in rng.choice(N_NEU, 3, replace=False):
            items.append({"kind": "neuron", "layer": layer, "unit": int(j),
                          "neuron": int(neuron_idx[layer, j]), "sign": int(rng.choice([1, -1]))})
        d = int(rng.integers(4))
        items.append({"kind": "random_dir", "layer": layer, "unit": N_NEU + d, "dir": d,
                      "sign": int(rng.choice([1, -1]))})
    order = rng.permutation(len(items))
    items = [items[i] for i in order]
    for k, it in enumerate(items):
        it["id"] = f"P{k + 1:03d}"
    return items


def fmt(ex, mx):
    return [f"- ({100 * a / mx:.0f}%) {snippet(w, p)}" for w, p, a in ex]


def build_explain():
    rng = np.random.default_rng(2)
    items = sample_items()
    os.makedirs(f"{S}/explain", exist_ok=True); os.makedirs(f"{S}/keys", exist_ok=True)
    for it in items:
        v = pole_values(it["layer"], it["unit"], it["sign"])
        top = per_doc_top(v, 40)
        mx = top[0][2]
        used = {doc_of[w] for w, _, _ in top}
        mid = band(v, 0.2, 0.5, 12, rng, used)
        used |= {doc_of[w] for w, _, _ in mid}
        low = band(v, 0.05, 0.2, 6, rng, used)
        rw = rng.integers(0, n_done, 4); rp = rng.integers(SKIP, shape[1], 4)
        rand = [(int(w), int(p), float(v[w, p - SKIP])) for w, p in zip(rw, rp)]
        it.update(max=mx, show_top=top[0::2][:20], held_top=top[1::2][:4], show_mid=mid[:8],
                  held_mid=mid[8:12], show_low=low, held_rand=rand,
                  frac_mass_above_20=float(v[v > 0.2 * mx].sum() / max(v[v > 0].sum(), 1e-9)))
    for b, chunk in enumerate(np.array_split(np.arange(len(items)), N_EXPLAIN_BATCHES)):
        lines = []
        for i in chunk:
            it = items[i]
            lines += [f"## {it['id']}", "", "Top activating examples (one per document, strongest first):"]
            lines += fmt(it["show_top"], it["max"])
            lines += ["", "Mid-range examples (20-50% of max):"] + fmt(it["show_mid"], it["max"])
            lines += ["", "Weak examples (5-20% of max):"] + fmt(it["show_low"], it["max"]) + [""]
        open(f"{S}/explain/batch_{b:02d}.md", "w").write("\n".join(lines))
    json.dump(items, open(f"{S}/keys/items.json", "w"))
    print(len(items), "items")


def build_judge():
    rng = np.random.default_rng(3)
    items = {it["id"]: it for it in json.load(open(f"{S}/keys/items.json"))}
    expl = {}
    for f in sorted(os.listdir(f"{S}/explain_out")):
        for r in json.load(open(f"{S}/explain_out/{f}")):
            expl[r["id"]] = r
    assert set(expl) == set(items), set(items) - set(expl)
    ids = list(items)
    pairs = [(i, i) for i in ids]
    for _ in range(N_CONTROL):
        a, b = rng.choice(ids, 2, replace=False)
        pairs.append((str(a), str(b)))
    pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    key = []
    for k, (e, s) in enumerate(pairs):
        it = items[s]
        snips = ([("top", x) for x in it["held_top"]] + [("mid", x) for x in it["held_mid"]] +
                 [("rand", x) for x in it["held_rand"]])
        snips = [snips[i] for i in rng.permutation(len(snips))]
        key.append({"jid": f"J{k + 1:03d}", "expl_of": e, "snips_of": s,
                    "snippets": [{"n": n + 1, "bin": b, "rel_act": a / it["max"],
                                  "text": snippet(w, p)} for n, (b, (w, p, a)) in enumerate(snips)]})
    os.makedirs(f"{S}/judge", exist_ok=True)
    for b, chunk in enumerate(np.array_split(np.arange(len(key)), N_JUDGE_BATCHES)):
        lines = []
        for i in chunk:
            j = key[i]
            lines += [f"## {j['jid']}", "", f"Explanation: {expl[j['expl_of']]['explanation']}", ""]
            lines += [f"{s['n']}. {s['text']}" for s in j["snippets"]] + [""]
        open(f"{S}/judge/batch_{b:02d}.md", "w").write("\n".join(lines))
    json.dump(key, open(f"{S}/keys/judge.json", "w"))
    print(len(key), "judge items")


{"explain": build_explain, "judge": build_judge}[sys.argv[1]]()
