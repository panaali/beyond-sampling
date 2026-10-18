"""Fixed coordinate projection; diagnostic arithmetic, no fit or recurrence."""
import numpy as np


def project(hidden, lookup, wo):
    h, z, w = (np.asarray(x, dtype=np.float64) for x in (hidden, lookup, wo))
    if h.ndim != 2 or z.shape != h.shape or w.ndim != 2:
        raise ValueError('two matching hidden matrices and one readout required')
    n, width = h.shape
    if n == 0 or width == 0 or w.shape[1] != width or w.shape[0] < 2:
        raise ValueError('nonempty batch/hidden and at least two classes required')
    if not all(np.isfinite(x).all() for x in (h, z, w)):
        raise ValueError('finite inputs required')
    if (h < 0).any():
        raise ValueError('actual hidden must be nonnegative')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        projected = np.maximum(z, 0.0)
        logits = projected @ w.T
        before = np.sum((z-h)**2, axis=1)
        after = np.sum((projected-h)**2, axis=1)
    if not all(np.isfinite(x).all() for x in (projected, logits, before, after)):
        raise ArithmeticError('nonfinite result')
    classes = w.shape[0]
    return dict(projected=projected, projected_logits=logits,
                sse_before=before, sse_after=after,
                named_expression_counts=dict(comparisons=n*width,
                    subtractions=2*n*width,
                    multiplications=2*n*width+n*width*classes,
                    additions=2*n*(width-1)+n*classes*(width-1)))
