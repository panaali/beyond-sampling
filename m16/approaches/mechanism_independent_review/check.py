"""T39 bounded, independent spectral/impulse/offset checks. No source imports."""
import time
START_CPU = time.process_time()
START_WALL = time.perf_counter()
import argparse
import hashlib
import json
from pathlib import Path
import resource
import numpy as np
import torch

EXPECTED = '498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path(__file__).with_name('results.json'))
    a = p.parse_args()
    digest = hashlib.sha256(a.weights.read_bytes()).hexdigest()
    assert digest == EXPECTED
    state = torch.load(a.weights, weights_only=True, map_location='cpu')
    wi = state['rnn.weight_ih_l0'].numpy()[:, 0].astype(float)
    wh = state['rnn.weight_hh_l0'].numpy().astype(float)
    wo = state['linear.weight'].numpy().astype(float)
    assert wi.shape == (16,) and wh.shape == (16, 16) and wo.shape == (10, 16)
    counts = dict(eigendecompositions=0, model_sequences=0, hidden_steps=0,
                  readout_vectors=0, least_squares_solves=0, random_inputs=0)

    def spectrum(mat):
        vals = np.linalg.eigvals(mat)
        counts['eigendecompositions'] += 1
        pairs = sorted((z for z in vals if z.imag > 1e-10), key=lambda z: np.angle(z))
        records = [dict(real=float(z.real), imag=float(z.imag), modulus=float(abs(z)),
                        angle_degrees=float(np.angle(z)*180/np.pi),
                        period=float(2*np.pi/np.angle(z))) for z in pairs]
        dominant = max(records, key=lambda x: x['modulus']) if records else None
        return dict(complex_pairs=records, dominant_complex_pair=dominant,
                    real_eigenvalues=sorted(float(z.real) for z in vals if abs(z.imag)<=1e-10))

    specs = {'raw': spectrum(wh)}
    for code in [64975, 63951, 65519, 62927, 65007]:
        active = np.flatnonzero(code & (1 << np.arange(16)))
        specs[str(code)] = dict(active=active.tolist(), **spectrum(wh[np.ix_(active, active)]))
    specs['subblock_10_11_12'] = spectrum(wh[np.ix_([10, 11, 12], [10, 11, 12])])
    pairs = [(m, s) for m in range(10) for s in range(10) if m != s]

    def forward(x):
        n = len(x)
        hs = np.zeros((n, 11, 16))
        masks = np.zeros((n, 10), dtype=int)
        for t in range(10):
            pre = hs[:, t] @ wh.T + x[:, t, None]*wi
            hs[:, t+1] = np.maximum(pre, 0)
            masks[:, t] = (pre > 0).astype(int) @ (1 << np.arange(16))
        logits = hs[:, 1:] @ wo.T
        counts['model_sequences'] += n
        counts['hidden_steps'] += 10*n
        counts['readout_vectors'] += 10*n
        return hs, masks, logits.argmax(axis=2)

    trajectories, impulse_records = {}, []
    for ratio in [.2, .8, .99]:
        x = np.zeros((90, 10))
        for i, (m, s) in enumerate(pairs):
            x[i, m], x[i, s] = 1., ratio
        hs, masks, pred = forward(x)
        correct = [int(pred[i, -1] == s) for i, (m, s) in enumerate(pairs)]
        countdown = []
        shifts = []
        for i, (m, s) in enumerate(pairs):
            for t in range(max(m, s), 10):
                expected = s+9-t
                countdown.append(dict(m=m, s=s, t=t, expected=expected,
                                      pred=int(pred[i, t]), correct=bool(pred[i,t]==expected)))
            for j in range(i):
                mm, ss = pairs[j]
                if s-m == ss-mm:
                    aa, bb = masks[i, max(m,s):], masks[j, max(mm,ss):]
                    shifts.append(bool(np.array_equal(aa[:min(len(aa),len(bb))], bb[:min(len(aa),len(bb))])))
        trajectories[ratio] = (hs, masks, pred)
        impulse_records.append(dict(ratio=ratio, correct=sum(correct), total=90,
            forward_correct=sum(correct[i] for i,(m,s) in enumerate(pairs) if m<s),
            reverse_correct=sum(correct[i] for i,(m,s) in enumerate(pairs) if m>s),
            failed_pairs=[list(pairs[i]) for i,c in enumerate(correct) if not c],
            predicts_max=sum(int(pred[i,-1]==m) for i,(m,s) in enumerate(pairs)),
            countdown_correct=sum(z['correct'] for z in countdown), countdown_total=len(countdown),
            countdown_failures=[z for z in countdown if not z['correct']],
            translation_comparisons=len(shifts), translation_failures=sum(not z for z in shifts)))
        if ratio == .8:
            scaled = forward(3*x)
            homogeneity = dict(scale=3, max_abs_hidden_residual=float(np.max(np.abs(scaled[0]-3*hs))),
                              gate_disagreements=int(np.sum(scaled[1]!=masks)),
                              prediction_disagreements=int(np.sum(scaled[2]!=pred)))
    ratio_gate_differences=[]
    for ratio in [.2,.99]:
        aa, bb = trajectories[ratio][1], trajectories[.8][1]
        post = [(i,t) for i,(m,s) in enumerate(pairs) for t in range(max(m,s),10)]
        ratio_gate_differences.append(dict(ratio=ratio, comparisons=len(post),
            differences=sum(int(aa[i,t]!=bb[i,t]) for i,t in post),
            examples=[dict(m=pairs[i][0],s=pairs[i][1],t=t,gate_ratio=int(aa[i,t]),gate_08=int(bb[i,t]))
                      for i,t in post if aa[i,t]!=bb[i,t]][:10]))

    design = np.zeros((90, 18))
    for row,(m,s) in enumerate(pairs):
        if m: design[row,m-1]=1
        if s: design[row,9+s-1]=1

    def audit_offset(delta):
        data = np.array([delta[m,s] for m,s in pairs])
        coef, _, rank, _ = np.linalg.lstsq(design, data, rcond=None)
        counts['least_squares_solves'] += 1
        f, g = np.zeros((10,data.shape[1])), np.zeros((10,data.shape[1]))
        f[1:], g[1:] = coef[:9], coef[9:]
        row_sums = delta.sum(axis=1)
        closed_f = (row_sums-row_sums[0])/10
        residual = data-design@coef
        ss_total = np.sum((data-data.mean(axis=0))**2)
        cycles = np.array([delta[i,j]+delta[j,k]+delta[k,i] for i in range(10)
                           for j in range(i+1,10) for k in range(j+1,10)])
        return dict(design_rank=int(rank), max_abs_antisymmetry=float(np.max(np.abs(f+g))),
                    max_abs_row_formula_error=float(np.max(np.abs(f-closed_f))),
                    r2=float(1-np.sum(residual**2)/ss_total),
                    max_abs_triangle_cycle=float(np.max(np.abs(cycles))),
                    max_abs_delta=float(np.max(np.abs(data))))

    final = np.zeros((10,10,16))
    for i,(m,s) in enumerate(pairs): final[m,s] = trajectories[.8][0][i,-1]
    delta = final-final.swapaxes(0,1)
    actual = audit_offset(delta)
    base=np.array([[[np.sin((k+1)*(i+2*j+1)) for k in range(3)] for j in range(10)] for i in range(10)])
    synthetic=audit_offset(base-base.swapaxes(0,1))
    sign_margins=np.array([(wo[s]-wo[m])@delta[m,s] for m,s in pairs])
    actual.update(positive_offset_discrimination_count=int(np.sum(sign_margins>0)),
                  min_offset_discrimination=float(sign_margins.min()))
    results=dict(checkpoint_sha256=digest, checkpoint_shape=dict(H=16,T=10),
                 array_sha256=hashlib.sha256(b''.join(v.tobytes() for v in (wi,wh,wo))).hexdigest(),
                 spectra=specs, impulses=impulse_records, homogeneity=homogeneity,
                 gate_ratio_comparisons=ratio_gate_differences,
                 offset_actual=actual, offset_synthetic=synthetic, counts=counts,
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 cpu_seconds_including_imports=time.process_time()-START_CPU,
                 wall_seconds=time.perf_counter()-START_WALL,
                 max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 numerical_scope='float64 diagnostics, not certified eigenvalue or trajectory bounds')
    a.out.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps({k:results[k] for k in ['offset_actual','offset_synthetic','counts','cpu_seconds_including_imports','max_rss_kib']},indent=2))


if __name__ == '__main__': main()
