"""T6: reproduction check, blinded explainer packets, and (after explanations) deduplicated judge packets.

usage: python build_t6.py check    -> compares stored activations with E1's study/keys/items.json
       python build_t6.py explain  -> essay_h2_rerun/explain/r{1,2}_pXX.md, data/t6/keys/items.json
       python build_t6.py judge    -> essay_h2_rerun/judge/jXX.md, data/t6/keys/judge.json
"""
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer

A, S, K = "data/t6", "essay_h2_rerun", "data/t6/keys"
SKIP, BEFORE, AFTER, NGRAM = 4, 24, 4, 8
N_NEU_PER_LAYER, N_DENSE_PER_LAYER, N_GAUSS_PER_LAYER = 10, 10, 2
N_NEURONS, N_DIRS = 32, 4
EXPLAIN_PACKET, JUDGE_PACKET, N_MISMATCH = 14, 20, 120

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
tokens = np.load(f"{A}/tokens.npy")
doc_of = np.load(f"{A}/doc_of.npy")
meta = json.load(open(f"{A}/meta.json"))
N_WIN, CTX = tokens.shape
assert json.load(open(f"{A}/progress.json"))["done"] == N_WIN
N_LAYERS = 28
mm = [np.memmap(f"{A}/L{L:02d}.f16", dtype=np.float16, mode="r", shape=(meta["U"], meta["T"]))
      for L in range(N_LAYERS)]
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


def ngrams(w, p):
    ids = tuple(int(t) for t in tokens[w][max(0, p - BEFORE):p + 1 + AFTER])
    return {ids[i:i + NGRAM] for i in range(len(ids) - NGRAM + 1)}


def pole_values(layer, unit, sign):
    return sign * np.asarray(mm[layer][unit], dtype=np.float32).reshape(N_WIN, CTX)[:, SKIP:]


def per_doc_top(v, k):
    flat = v.reshape(-1)
    order = np.argpartition(-flat, 20000)[:20000]
    order = order[np.argsort(-flat[order])]
    out, seen = [], set()
    for f in order:
        w, p = divmod(int(f), v.shape[1])
        if doc_of[w] in seen:
            continue
        seen.add(doc_of[w]); out.append((w, p + SKIP, float(flat[f])))
        if len(out) == k:
            break
    return out


def band(v, lo, hi, k, rng, exclude_docs, mx):
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


def check():
    e1 = json.load(open("study/keys/items.json"))
    worst, n = 0.0, 0
    for it in e1:
        unit = it["unit"]  # neurons 0..31 and Gaussian dirs 32..35: same layout as here
        v = pole_values(it["layer"], unit, it["sign"])
        for key in ["show_top", "held_top", "show_mid", "held_mid", "show_low", "held_rand"]:
            for w, p, a in it[key]:
                err = abs(float(v[w, p - SKIP]) - a) / it["max"]
                worst = max(worst, err); n += 1
    print(f"E1 reproduction: {n} stored activations, worst |diff|/max = {worst:.4f}")
    json.dump({"n": n, "worst_rel_err": worst, "pass": worst <= 0.01}, open(f"{S}/repro_check.json", "w"))


def sample_items():
    rng = np.random.default_rng(11)
    items = []
    for L in range(N_LAYERS):
        for j in rng.choice(N_NEURONS, N_NEU_PER_LAYER, replace=False):
            items.append({"kind": "neuron", "layer": L, "unit": int(j), "sign": int(rng.choice([1, -1]))})
        for d in rng.choice(N_DIRS, N_GAUSS_PER_LAYER, replace=False):
            items.append({"kind": "gauss", "layer": L, "unit": N_NEURONS + int(d), "sign": int(rng.choice([1, -1]))})
        for d in range(N_DENSE_PER_LAYER):
            items.append({"kind": "dense", "layer": L, "unit": N_NEURONS + N_DIRS + d,
                          "sign": int(rng.choice([1, -1]))})
    items = [items[i] for i in rng.permutation(len(items))]
    ids = rng.choice(np.arange(1000, 10000), len(items), replace=False)
    for it, i in zip(items, ids):
        it["id"] = f"U{i}"
    return items


def fmt(ex, mx):
    return [f"- ({100 * a / mx:.0f}%) {snippet(w, p)}" for w, p, a in ex]


def dedupe(cands, banned, k):
    kept = []
    for c in cands:
        g = ngrams(c[0], c[1])
        if g & banned:
            continue
        kept.append(c); banned |= g
        if len(kept) == k:
            break
    return kept


def build_explain():
    rng = np.random.default_rng(12)
    items = sample_items()
    os.makedirs(f"{S}/explain", exist_ok=True); os.makedirs(K, exist_ok=True)
    for n, it in enumerate(items):
        v = pole_values(it["layer"], it["unit"], it["sign"])
        top = per_doc_top(v, 80)
        mx = top[0][2]
        used = {doc_of[w] for w, _, _ in top}
        mid = band(v, 0.2, 0.5, 40, rng, used, mx)
        used |= {doc_of[w] for w, _, _ in mid}
        low = band(v, 0.05, 0.2, 6, rng, used, mx)
        rw = rng.integers(0, N_WIN, 60); rp = rng.integers(SKIP, CTX, 60)
        rand = [(int(w), int(p), float(v[w, p - SKIP])) for w, p in zip(rw, rp)]
        show_top, show_mid = top[0:40:2], mid[:8]
        banned = set().union(*(ngrams(w, p) for w, p, _ in show_top + show_mid + low))
        held_top = dedupe(top[1::2], banned, 4)
        held_mid = dedupe(mid[8:], banned, 4)
        held_rand = dedupe(rand, banned, 4)
        assert len(held_top) == 4 and len(held_rand) == 4, it
        it.update(max=mx, show_top=show_top, show_mid=show_mid, show_low=low, held_top=held_top,
                  held_mid=held_mid, held_rand=held_rand,
                  n_dropped_top=top[1::2].index(held_top[-1]) + 1 - 4,
                  top_token_share=float(np.mean([tokens[w][p] == tokens[top[0][0]][top[0][1]] for w, p, _ in top])),
                  frac_mass_above_20=float(v[v > 0.2 * mx].sum() / max(v[v > 0].sum(), 1e-9)))
        if n % 50 == 0:
            print(n, flush=True)
    packets = []
    for r in (1, 2):
        order = rng.permutation(len(items))
        for b, chunk in enumerate(np.array_split(order, len(items) // EXPLAIN_PACKET)):
            name = f"r{r}_p{b:02d}"
            lines = []
            for i in chunk:
                it = items[i]
                lines += [f"## {it['id']}", "", "Top activating examples (one per document, strongest first):"]
                lines += fmt(it["show_top"], it["max"])
                lines += ["", "Mid-range examples (20-50% of max):"] + fmt(it["show_mid"], it["max"])
                lines += ["", "Weak examples (5-20% of max):"] + fmt(it["show_low"], it["max"]) + [""]
            open(f"{S}/explain/{name}.md", "w").write("\n".join(lines))
            packets.append({"packet": name, "round": r, "ids": [items[i]["id"] for i in chunk]})
    json.dump(items, open(f"{K}/items.json", "w"))
    json.dump(packets, open(f"{K}/explain_packets.json", "w"))
    print(len(items), "items,", len(packets), "explainer packets")


def build_judge():
    rng = np.random.default_rng(13)
    items = {it["id"]: it for it in json.load(open(f"{K}/items.json"))}
    expl = {}
    for f in sorted(os.listdir(f"{S}/explain_out")):
        r = int(f[1])
        for x in json.load(open(f"{S}/explain_out/{f}")):
            expl[(x["id"], r)] = x
    assert set(expl) == {(i, r) for i in items for r in (1, 2)}, len(expl)
    ids = list(items)
    pairs = [((i, r), i) for i in ids for r in (1, 2)]
    for _ in range(N_MISMATCH):
        a, b = rng.choice(ids, 2, replace=False)
        pairs.append(((str(a), int(rng.integers(1, 3))), str(b)))
    pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    n_pk = -(-len(pairs) // JUDGE_PACKET)
    bins = [[] for _ in range(n_pk)]
    for pr in pairs:  # no packet holds two items with the same snippets
        for b in sorted(range(n_pk), key=lambda b: len(bins[b])):
            if len(bins[b]) < JUDGE_PACKET and all(q[1] != pr[1] for q in bins[b]):
                bins[b].append(pr); break
        else:
            raise RuntimeError("packing failed")
    key, k = [], 0
    os.makedirs(f"{S}/judge", exist_ok=True)
    for b, chunk in enumerate(bins):
        lines = []
        for (e, s) in chunk:
            k += 1
            it = items[s]
            sn = ([("top", x) for x in it["held_top"]] + [("mid", x) for x in it["held_mid"]] +
                  [("rand", x) for x in it["held_rand"]])
            sn = [sn[i] for i in rng.permutation(len(sn))]
            jid = f"J{k:04d}"
            ex = expl[e]["explanation"]
            rows = []
            for n, (bn, (w, p, a)) in enumerate(sn):
                t = tstr(tokens[w][p]).strip().lower()
                rows.append({"n": n + 1, "bin": bn, "rel_act": a / it["max"], "text": snippet(w, p),
                             "lexical": len(t) >= 2 and t in ex.lower()})
            key.append({"jid": jid, "packet": f"j{b:02d}", "expl_of": list(e), "snips_of": s, "snippets": rows})
            lines += [f"## {jid}", "", f"Explanation: {ex}", ""] + [f"{x['n']}. {x['text']}" for x in rows] + [""]
        open(f"{S}/judge/j{b:02d}.md", "w").write("\n".join(lines))
    json.dump(key, open(f"{K}/judge.json", "w"))
    print(len(key), "judge items,", len(bins), "judge packets")


{"check": check, "explain": build_explain, "judge": build_judge}[sys.argv[1]]()
