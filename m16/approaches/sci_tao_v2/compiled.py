"""Optional ctypes backend. Build with `python .../sci_tao_v2/build.py`."""
import ctypes
from pathlib import Path
import numpy as np

COUNTERS = ('prefix_steps', 'suffix_steps', 'readout_vectors', 'root_vectors',
            'envelope_sides', 'cdf_calls', 'recurrence_flops', 'root_flops',
            'readout_flops', 'envelope_flops', 'bookkeeping_flops')
_library = None


def _load():
    global _library
    if _library is None:
        path = Path(__file__).with_name('_walker.so')
        if not path.exists():
            raise RuntimeError('C++ backend absent; run python m16/approaches/sci_tao_v2/build.py')
        _library = ctypes.CDLL(str(path))
        d = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        i = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
        _library.walk_segments.argtypes = [ctypes.c_int]*3 + [d]*4 + [i]*2 + [d, ctypes.c_int,
                                              ctypes.c_double, ctypes.c_int, d, i, i]
        _library.walk_segments.restype = ctypes.c_int
    return _library


def walk_mass(model, y, a, b, tau, kind, eps=1e-10, max_cells=2000, details=False):
    Wi, Wh, Wo = (np.ascontiguousarray(w, dtype=np.float64) for w in model)
    y = np.ascontiguousarray(y, dtype=np.float64)
    if y.ndim != 2 or Wi.ndim != 1:
        raise ValueError('Expected matrix face inputs and vector input weights')
    n, T = y.shape
    H = len(Wi)
    if H<1 or T<2 or Wh.shape!=(H,H) or Wo.shape!=(T,H) or kind not in ('12','23'):
        raise ValueError('Expected bias-free ReLU (Wi[H], Wh[H,H], Wo[T,H]), T>=2, kind 12/23')
    a, b = (np.ascontiguousarray(v, dtype=np.int64) for v in (a,b))
    tau = np.ascontiguousarray(tau, dtype=np.float64)
    if any(v.shape != (n,) for v in (a,b,tau)) or np.any(a<0) or np.any(a>=T) or np.any(b<0) or np.any(b>=T) or np.any(a==b):
        raise ValueError('Expected distinct in-range coordinate vectors and one window per row')
    if np.any(tau<0) or not all(np.isfinite(v).all() for v in (Wi,Wh,Wo,y,tau)):
        raise ValueError('Finite weights/points and finite nonnegative windows required')
    if not np.isfinite(eps) or eps<0 or max_cells<0:
        raise ValueError('eps and max_cells must be nonnegative')
    mass = np.zeros(n)
    cells = np.zeros(n, dtype=np.int64)
    counts = np.zeros((n,len(COUNTERS)), dtype=np.int64)
    status = _load().walk_segments(n,H,T,Wi,Wh,Wo,y,a,b,tau,int(kind),eps,max_cells,mass,cells,counts)
    if status==1:
        raise RuntimeError(f'max_cells={max_cells} exhausted; no partial integral returned')
    if status==2:
        raise ArithmeticError('Segment walker failed to advance')
    if status==3:
        raise FloatingPointError('Nonfinite state or readout in compiled walker')
    if status:
        raise RuntimeError(f'Unexpected compiled walker status {status}')
    unit = 6000.*(T*(2*H*H+2*H)+2*H*T)/5760.
    result = (mass,counts[:,6:].sum(axis=1)/unit,cells)
    return result + ({key:counts[:,j] for j,key in enumerate(COUNTERS)},) if details else result
