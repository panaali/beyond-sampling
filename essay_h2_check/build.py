"""T383 item building (own code). All steps before `judge` are LLM-free.

usage:
  python build.py select   choose the control construction (pre-registered rule), verify the density gate by
                           re-forwarding the final controls' 12 strongest shown examples, match the texture-matched
                           neuron arm, check balance -> essay_h2_check/choice.json, data/t383/keys/select.json
  python build.py explain  -> essay_h2_check/explain/r{1,2}_pNN.md, data/t383/keys/items.json, explain_packets.json
  python build.py judge    (after all explainer outputs exist) -> essay_h2_check/judge/jNN.md, keys/judge.json
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

A = os.environ.get("T383_OUT", "data/t383")
S, K = "essay_h2_check", f"{A}/keys"
SKIP, BEFORE, AFTER, NGRAM = 4, 24, 4, 8
NL, DM, N_PER_LAYER = 28, 3072, 11
SEL_MAX, GATE_MAX, CALIPER = 0.13, 0.15, 0.05
EXPLAIN_PACKET, JUDGE_PACKET, N_MISMATCH = 21, 24, 120

meta = json.load(open(f"{A}/meta.json"))
tokens, doc_of = np.load(f"{A}/tokens.npy"), np.load(f"{A}/doc_of.npy")
N_WIN, CTX = tokens.shape
U, T = meta["U"], meta["T"]
N_NEU = len(meta["layers"][0]["plain"]) + len(meta["layers"][0]["pool"])
mm = [np.memmap(f"{A}/L{L:02d}.f16", dtype=np.float16, mode="r", shape=(U, T)) for L in range(NL)]


def pole(L, u, s):
    """Pole values (N_WIN, CTX - SKIP), float32."""
    return s * np.asarray(mm[L][u], dtype=np.float32).reshape(N_WIN, CTX)[:, SKIP:]


def per_doc_top(val, pos, k):
    """Best window per document, strongest first: list of (window, position, value)."""
    order = np.argsort(-val, kind="stable")
    _, first = np.unique(doc_of[order], return_index=True)
    best = order[np.sort(first)][:k]
    return [(int(w), int(pos[w]), float(val[w])) for w in best]


def top_of(v, k):
    return per_doc_top(v.max(1), v.argmax(1) + SKIP, k)


def texture(v):
    mx = v.max()
    return float(v[v > 0.2 * mx].sum() / max(v[v > 0].sum(), 1e-12))


def unit_ref(kind, L, j):
    """(stored unit index, sign, neuron id or -1) for plain/pool neuron j or candidate direction j."""
    m = meta["layers"][L]
    if kind == "plain":
        return j, m["plain_sign"][j], m["plain"][j]
    if kind == "pool":
        return len(m["plain"]) + j, m["pool_sign"][j], m["pool"][j]
    return N_NEU + j, m["dir_sign"][j], -1


def select():
    import torch
    from transformers import AutoModel
    os.makedirs(K, exist_ok=True)
    rec = np.load(f"{A}/winmax.npz")
    assert int(rec["done"]) == N_WIN
    nd, nsel = meta["n_dir"], meta["n_sel"]
    table = {}
    for ci, c in enumerate(meta["constructions"]):  # selection half: directions 0-7 of each construction
        sh = []
        for L in range(NL):
            for j in range(ci * nd, ci * nd + nsel):
                top = per_doc_top(rec["val"][:, L, j], rec["pos"][:, L, j], 24)
                sh.append(np.median([rec["top1"][w, L, j] for w, _, _ in top[0:24:2]]))
        table[c] = {"selection_median_top1": float(np.median(sh)), "n": len(sh)}
        print(c, table[c], flush=True)
    order = meta["constructions"]
    passing = [c for c in order if table[c]["selection_median_top1"] <= SEL_MAX]
    if passing:
        choice, why = passing[0], f"first in order with selection median <= {SEL_MAX}"
    else:
        c = min(order, key=lambda c: table[c]["selection_median_top1"])
        if table[c]["selection_median_top1"] > GATE_MAX:
            json.dump({"table": table, "choice": None}, open(f"{S}/choice.json", "w"), indent=1)
            sys.exit("no construction <= 0.15 on the selection half: stop and amend (README)")
        choice, why = c, f"none <= {SEL_MAX}; lowest, <= {GATE_MAX}"
    ci = order.index(choice)
    rng = np.random.default_rng(3831)
    controls = [("dir", L, j) for L in range(NL) for j in range(ci * nd + nsel, ci * nd + nd)]
    assert len(controls) == NL * N_PER_LAYER
    # definitive gate: re-forward the 12 strongest shown examples (top[0:24:2]) of every final control
    dirs = np.load(f"{A}/dirs.npy")
    model = AutoModel.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32, attn_implementation="sdpa").eval()
    acts = {}
    for i, layer in enumerate(model.layers):
        layer.mlp.down_proj.register_forward_pre_hook(lambda m, inp, i=i: acts.__setitem__(i, inp[0].detach()))
    need = {}
    shown = {}
    for (_, L, j) in controls:
        u, s, _ = unit_ref("dir", L, j)
        shown[(L, j)] = top_of(pole(L, u, s), 24)[0:24:2]
        for w, p, a in shown[(L, j)]:
            need.setdefault(w, []).append((L, j, p, a))
    res = {k: [] for k in shown}
    ws = sorted(need)
    for b in range(0, len(ws), 16):
        bw = ws[b:b + 16]
        with torch.inference_mode():
            model(torch.tensor(tokens[bw], dtype=torch.long))
        for r, w in enumerate(bw):
            for L, j, p, a in need[w]:
                s = meta["layers"][L]["dir_sign"][j]
                c = s * acts[L][r, p].numpy() * dirs[L, :, j]
                res[(L, j)].append((float(c.max() / c.sum()), float(np.sort(c)[-5:].sum() / c.sum()),
                                    float(abs(c.sum() - a) / max(abs(a), 1e-9))))
        if b % 320 == 0:
            print("gate forward", b, len(ws), flush=True)
    per_item = {k: (float(np.median([x[0] for x in v])), float(np.median([x[1] for x in v]))) for k, v in res.items()}
    gate = float(np.median([x[0] for x in per_item.values()]))
    worst_err = float(max(x[2] for v in res.values() for x in v))
    gate_ok = gate <= GATE_MAX
    print("gate median top-1", gate, "pass", gate_ok, "worst rel proj err", worst_err, flush=True)
    # texture matching: per layer, optimal 1:1 assignment of the 11 controls to 11 distinct pool neurons
    ctex = {(L, j): texture(pole(L, *unit_ref("dir", L, j)[:2])) for (_, L, j) in controls}
    pairs = []
    for L in range(NL):
        npool = len(meta["layers"][L]["pool"])
        ptex = np.array([texture(pole(L, *unit_ref("pool", L, q)[:2])) for q in range(npool)])
        cj = [j for (_, LL, j) in controls if LL == L]
        cost = np.abs(np.array([ctex[(L, j)] for j in cj])[:, None] - ptex[None])
        r, q = linear_sum_assignment(cost)
        for a, b in zip(r, q):
            pairs.append({"layer": L, "control": int(cj[a]), "pool": int(b), "control_tex": ctex[(L, cj[a])],
                          "matched_tex": float(ptex[b]), "absdiff": float(cost[a, b])})
        print("match layer", L, flush=True)
    ct, mt = np.array([p["control_tex"] for p in pairs]), np.array([p["matched_tex"] for p in pairs])
    within = np.array([p["absdiff"] <= CALIPER for p in pairs])
    balance = {"median_control_tex": float(np.median(ct)), "median_matched_tex": float(np.median(mt)),
               "median_gap": float(abs(np.median(ct) - np.median(mt))), "frac_within_caliper": float(within.mean()),
               "n_within_caliper": int(within.sum())}
    balance["pass"] = balance["median_gap"] <= 0.03 and balance["frac_within_caliper"] >= 0.80
    plain_tex = [texture(pole(L, *unit_ref("plain", L, j)[:2])) for L in range(NL)
                 for j in range(len(meta["layers"][L]["plain"]))]
    out = {"selection_table": table, "choice": choice, "why": why,
           "gate": {"median_top1": gate, "median_top5": float(np.median([x[1] for x in per_item.values()])),
                    "frac_items_le_0.15": float(np.mean([x[0] <= GATE_MAX for x in per_item.values()])),
                    "worst_rel_proj_err": worst_err, "pass": gate_ok, "n_items": len(per_item)},
           "balance": balance, "median_plain_tex": float(np.median(plain_tex))}
    json.dump(out, open(f"{S}/choice.json", "w"), indent=1)
    json.dump({"controls": [list(c) for c in controls], "pairs": pairs,
               "gate_items": {f"{L}_{j}": v for (L, j), v in per_item.items()}}, open(f"{K}/select.json", "w"))
    print(json.dumps(out, indent=1))


def tstr(t, _c={}):
    if t not in _c:
        _c[t] = TOK.decode([int(t)]).replace("\n", "⏎")
    return _c[t]


def snippet(w, p):
    ids = tokens[w]
    return ("".join(tstr(t) for t in ids[max(0, p - BEFORE):p]) + f"[[{tstr(ids[p])}]]" +
            "".join(tstr(t) for t in ids[p + 1:p + 1 + AFTER]))


def grams(w, p):
    ids = tuple(int(t) for t in tokens[w][max(0, p - BEFORE):p + 1 + AFTER])
    return {ids[i:i + NGRAM] for i in range(len(ids) - NGRAM + 1)}


def band(v, lo, hi, k, rng, used):
    ws, ps = np.nonzero((v >= lo) & (v < hi))
    out, seen = [], set(used)
    for i in rng.permutation(len(ws)):
        w = int(ws[i])
        if doc_of[w] in seen:
            continue
        seen.add(doc_of[w])
        out.append((w, int(ps[i]) + SKIP, float(v[w, ps[i]])))
        if len(out) == k:
            break
    return out


def dedupe(cands, banned, k):
    kept = []
    for c in cands:
        g = grams(c[0], c[1])
        if g & banned:
            continue
        kept.append(c)
        banned |= g
        if len(kept) == k:
            break
    return kept


def label(ex, mx):
    return [f"- ({100 * a / mx:.0f}%) {snippet(w, p)}" for w, p, a in ex]


def explain():
    sel = json.load(open(f"{K}/select.json"))
    rng = np.random.default_rng(3832)
    units = [("plain", L, j) for L in range(NL) for j in range(len(meta["layers"][L]["plain"]))]
    units += [("matched", p["layer"], p["pool"]) for p in sel["pairs"]]
    units += [("control", L, j) for _, L, j in sel["controls"]]
    items = []
    ids = rng.choice(np.arange(1000, 10000), len(units), replace=False)
    for n, (arm, L, j) in enumerate(units):
        u, s, neuron = unit_ref({"plain": "plain", "matched": "pool", "control": "dir"}[arm], L, j)
        v = pole(L, u, s)
        top = top_of(v, 80)
        mx = top[0][2]
        used = {doc_of[w] for w, _, _ in top}
        mid = band(v, 0.2 * mx, 0.5 * mx, 40, rng, used)
        used |= {doc_of[w] for w, _, _ in mid}
        low = band(v, 0.05 * mx, 0.2 * mx, 6, rng, used)
        rw, rp = rng.integers(0, N_WIN, 60), rng.integers(SKIP, CTX, 60)
        rand = [(int(w), int(p), float(v[w, p - SKIP])) for w, p in zip(rw, rp)]
        show_top, show_mid = top[0:40:2], mid[:8]
        banned = set().union(*(grams(w, p) for w, p, _ in show_top + show_mid + low))
        held_top, held_mid, held_rand = dedupe(top[1::2], banned, 4), dedupe(mid[8:], banned, 4), dedupe(rand, banned, 4)
        assert len(held_top) == 4 and len(held_rand) == 4, (arm, L, j)
        t0 = tokens[top[0][0]][top[0][1]]
        items.append({"id": f"U{ids[n]}", "arm": arm, "layer": L, "unit": int(u), "j": int(j), "neuron": int(neuron),
                      "sign": int(s), "max": mx, "show_top": show_top, "show_mid": show_mid, "show_low": low,
                      "held_top": held_top, "held_mid": held_mid, "held_rand": held_rand,
                      "n_dropped_top": top[1::2].index(held_top[-1]) + 1 - 4,
                      "texture": texture(v), "flat20": show_top[-1][2] / mx,
                      "top_token_share": float(np.mean([tokens[w][p] == t0 for w, p, _ in top]))})
        if n % 50 == 0:
            print("item", n, flush=True)
    os.makedirs(f"{S}/explain", exist_ok=True)
    packets = []
    for r in (1, 2):
        perm = rng.permutation(len(items))
        for b, chunk in enumerate(np.array_split(perm, len(items) // EXPLAIN_PACKET)):
            name = f"r{r}_p{b:02d}"
            lines = []
            for i in chunk:
                it = items[i]
                lines += [f"## {it['id']}", "", "Top activating examples (one per document, strongest first):"]
                lines += label(it["show_top"], it["max"])
                lines += ["", "Mid-range examples (20-50% of max):"] + label(it["show_mid"], it["max"])
                lines += ["", "Weak examples (5-20% of max):"] + label(it["show_low"], it["max"]) + [""]
            open(f"{S}/explain/{name}.md", "w").write("\n".join(lines))
            packets.append({"packet": name, "round": r, "ids": [items[i]["id"] for i in chunk]})
    json.dump(items, open(f"{K}/items.json", "w"))
    json.dump(packets, open(f"{K}/explain_packets.json", "w"))
    print(len(items), "items,", len(packets), "explainer packets")


def judge():
    rng = np.random.default_rng(3833)
    items = {it["id"]: it for it in json.load(open(f"{K}/items.json"))}
    expl = {}
    for f in sorted(os.listdir(f"{S}/explain_out")):
        for x in json.load(open(f"{S}/explain_out/{f}")):
            expl[(x["id"], int(f[1]))] = x
    assert set(expl) == {(i, r) for i in items for r in (1, 2)}, len(expl)
    ids = sorted(items)
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
                bins[b].append(pr)
                break
        else:
            raise RuntimeError("packing failed")
    key, k = [], 0
    os.makedirs(f"{S}/judge", exist_ok=True)
    for b, chunk in enumerate(bins):
        lines = []
        for e, s in chunk:
            k += 1
            it = items[s]
            sn = [("top", x) for x in it["held_top"]] + [("mid", x) for x in it["held_mid"]] + \
                 [("rand", x) for x in it["held_rand"]]
            sn = [sn[i] for i in rng.permutation(len(sn))]
            ex = expl[e]["explanation"]
            rows = []
            for n, (bn, (w, p, a)) in enumerate(sn):
                t = tstr(tokens[w][p]).strip().lower()
                rows.append({"n": n + 1, "bin": bn, "rel_act": a / it["max"], "text": snippet(w, p),
                             "lexical": len(t) >= 2 and t in ex.lower()})
            jid = f"J{k:04d}"
            key.append({"jid": jid, "packet": f"j{b:02d}", "expl_of": list(e), "snips_of": s, "snippets": rows})
            lines += [f"## {jid}", "", f"Explanation: {ex}", ""] + [f"{x['n']}. {x['text']}" for x in rows] + [""]
        open(f"{S}/judge/j{b:02d}.md", "w").write("\n".join(lines))
    json.dump(key, open(f"{K}/judge.json", "w"))
    print(len(key), "judge items,", len(bins), "judge packets")


if __name__ == "__main__":
    if sys.argv[1] in ("explain", "judge"):
        from transformers import AutoTokenizer
        TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    {"select": select, "explain": explain, "judge": judge}[sys.argv[1]]()
