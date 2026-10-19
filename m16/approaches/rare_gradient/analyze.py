"""T372 mechanistic reading: top weights and per-neuron elasticities of P(E1) from the unbiased line-CMC gradient
(secondary; the one that passed on s526) and from the K3 gradient (primary), their agreement, and the T241 roles.
Elasticity of weight w: e_w = w dP/dw / P = d log P / d log w. Neuron i: E_in(i) = sum of e_w over weights INTO i
(Wi[i], Wh[i, :]) = d log P / d log(gain of neuron i's pre-activation); E_out(i) = over weights OUT of i (Wh[:, i],
Wo[:, i]) = d log P / d log(gain of neuron i's output). usage: python analyze.py -> results/mech_<model>.json, mech.txt"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROLE = {2: "running max", 4: "record detector", 1: "x9 comparator", 6: "x9 comparator", 7: "x9 comparator", 8: "x9 comparator",
        0: "clock", 10: "clock", 11: "clock", 12: "clock", 14: "clock", 15: "clock (4.32 mode)", 5: "late-x8", 9: "late-x8",
        3: "4.32 mode", 13: "4.32 mode"}                                           # T241 / FINDINGS (M_16,10 only)


def names():
    out = [f"Wi[n{i}]" for i in range(16)] + [f"Wh[n{i}<-n{j}]" for i in range(16) for j in range(16)]
    return out + [f"Wo[o{c}<-n{j}]" for c in range(10) for j in range(16)]


def neuron_sums(e):
    Wi, Wh, Wo = e[:16], e[16:272].reshape(16, 16), e[272:].reshape(10, 16)
    return Wi + Wh.sum(1), Wh.sum(0) + Wo.sum(0)


def main():
    NM = names(); txt = []
    for nm in ("orig", "dev3_s526", "dev3_s528"):
        fd, fc = (os.path.join(HERE, "results", f"{p}_{nm}.json") for p in ("dirs", "cmc"))
        if not (os.path.exists(fd) and os.path.exists(fc)):
            continue
        D, C = json.load(open(fd)), json.load(open(fc)); w = np.array(D["w"]); P = C["P"]
        gc, sc, gk = np.array(C["grad"]), np.array(C["grad_se"]), np.array(D["grad"])
        cos = float(gc @ gk / np.linalg.norm(gc) / np.linalg.norm(gk)); scale = float(gk @ gc / (gc @ gc))
        ec, ek = w * gc / P, w * gk / D["est"]; ec_se = np.abs(w) * sc / P
        o = np.argsort(-np.abs(gc))[:15]; oe = np.argsort(-np.abs(ec))[:15]
        Ein, Eout = neuron_sums(ec); Kin, Kout = neuron_sums(ek)
        # s.e. of neuron sums is not available from per-weight s.e. (correlated); report bootstrap-free z of top weights only
        txt.append(f"== {nm} E1({D['t0']}): P {P:.4e}; |dlogP/dW| CMC {np.linalg.norm(gc) / P:.1f} vs K3 {np.linalg.norm(gk) / D['est']:.1f}; "
                   f"cos(K3, CMC) {cos:.3f}; K3 = {scale:.3f} x CMC along CMC; weights with |g| > 3 s.e.: {(np.abs(gc) > 3 * sc).sum()}/432")
        txt.append("  top |dlogP/dw| (CMC):  " + ", ".join(f"{NM[i]} {gc[i] / P:+.1f}+-{sc[i] / P:.1f} (K3 {gk[i] / D['est']:+.1f})" for i in o[:10]))
        txt.append("  top elasticity w dlogP/dw (CMC): " + ", ".join(f"{NM[i]} {ec[i]:+.1f}+-{ec_se[i]:.1f}" for i in oe[:10]))
        txt.append("  neuron E_in / E_out (CMC; K3 in brackets):")
        for i in np.argsort(-(np.abs(Ein) + np.abs(Eout))):
            txt.append(f"    n{i:<2} {ROLE.get(i, '') if nm == 'orig' else '':18s} in {Ein[i]:+8.1f} ({Kin[i]:+7.1f})  out {Eout[i]:+8.1f} ({Kout[i]:+7.1f})")
        out_rows = dict(model=nm, P=P, cos=cos, scale=scale, top_grad=[(NM[i], gc[i] / P, sc[i] / P, gk[i] / D["est"]) for i in o],
                        top_elast=[(NM[i], ec[i], ec_se[i], ek[i]) for i in oe], E_in=Ein.tolist(), E_out=Eout.tolist(), K3_E_in=Kin.tolist(),
                        K3_E_out=Kout.tolist(), W_block_norms=dict(Wi=float(np.linalg.norm(gc[:16]) / P), Wh=float(np.linalg.norm(gc[16:272]) / P),
                                                                     Wo=float(np.linalg.norm(gc[272:]) / P)))
        json.dump(out_rows, open(os.path.join(HERE, "results", f"mech_{nm}.json"), "w"), indent=1)
    open(os.path.join(HERE, "results", "mech.txt"), "w").write("\n".join(txt) + "\n"); print("\n".join(txt))


if __name__ == "__main__":
    main()
