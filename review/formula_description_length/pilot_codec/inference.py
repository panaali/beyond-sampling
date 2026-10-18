"""Inference wrapper appended to the unchanged reviewed feature definitions.

NumPy and Python numeric semantics are the common T25 conditioning. The framed
AST includes the feature functions and this wrapper; no fitting code is needed.
"""

def table_predict(cells, design, record):
    ids, pooled, coefficients = record
    pooled = np.asarray(pooled, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    answer = design @ pooled
    for i, cell in enumerate(ids):
        keep = cells == cell
        answer[keep] = design[keep] @ coefficients[i]
    return answer


def predict(x, payload):
    target, version, width, steps, banks, finals, readout = payload
    if version != 1 or width != 16 or steps != 10 or target not in ('hidden', 'labels'):
        raise ValueError('payload header')
    n = len(x)
    hinges = np.zeros((n, 11, 4), dtype=np.float32)
    for step, ids, pooled, coefficients in banks:
        cells, design = features(x, step, 5, hinges)
        pre = table_predict(cells, design, (ids, pooled, coefficients))
        hinges[:, step, :] = np.maximum(pre, 0).astype(np.float32)
        if not np.isfinite(hinges[:, step, :]).all():
            raise ValueError('nonfinite bank')
    chosen = np.zeros((n, 16), dtype=np.float32)
    covered = []
    for family, outputs, ids, pooled, coefficients in finals:
        cells, design = features(x, 10, family, hinges)
        # Preserve all16 output columns for selected families: selecting columns
        # before BLAS would change its shape and need fresh numerical parity.
        pre = table_predict(cells, design, (ids, pooled, coefficients))
        stored = pre.astype(np.float32)
        if not np.isfinite(stored).all():
            raise ValueError('nonfinite final table')
        for j in outputs:
            chosen[:, j] = stored[:, j]
            covered.append(j)
    if sorted(covered) != list(range(16)):
        raise ValueError('output partition')
    hidden = np.maximum(chosen.astype(np.float64), 0)
    if target == 'hidden':
        if readout is not None:
            raise ValueError('conditional code must omit readout')
        return hidden
    wo = np.asarray(readout, dtype=np.float64)
    if wo.shape != (10, 16) or not np.isfinite(wo).all():
        raise ValueError('readout')
    logits = hidden @ wo.T
    if not np.isfinite(logits).all():
        raise ValueError('nonfinite logits')
    return logits.argmax(axis=1)
