"""Frozen allocation/layout gate only; no matrix products or source mass calls."""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import resource
import time
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
RAYS = Path('/workspace/work/recover-small_h2_910/m16/approaches/leaf_mass_review/error_captured.npz')
RAYS_SHA = '15f9e5e483c69325c10fcbe4eaaa80f91503b177070ccec1ff2569a11c9436b6'
SOURCE_SHA = 'b8d3baf0d85d47bf4a7a204f82cdfc648de580b47baea5f42a4c77c49beb906a'

def layout(a):
    return dict(shape=list(a.shape), strides=list(a.strides), C=bool(a.flags.c_contiguous),
                F=bool(a.flags.f_contiguous), dtype=str(a.dtype))

def main():
    out = HERE / 'layout_applicability.json'
    assert not out.exists(), 'Preserve existing applicability evidence'
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    start = time.process_time()
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for key, path in manifest['binary_paths'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == manifest['binary_sha256'][key]
    assert hashlib.sha256(RAYS.read_bytes()).hexdigest() == RAYS_SHA
    with zipfile.ZipFile(RAYS) as z:
        with z.open('rays.npy') as f:
            version = np.lib.format.read_magic(f)
            assert version == (1, 0)
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    assert shape == (35098, 4, 4) and not fortran and dtype == np.dtype('float64')
    indexed = dict(shape=[4, 4], strides=[32, 8], C=True, F=False, dtype='float64')
    source = (HERE.parent / 'frozen/estT_snapshot.py').read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == SOURCE_SHA
    funcs = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_duffy']
    assert len(funcs) == 1
    env = {'np': np}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), '<frozen _duffy only>', 'exec'), env)
    du, dw = env['_duffy'](10)
    assert du.shape == (1000, 4) and du.strides == (32, 8) and du.flags.c_contiguous
    # Allocation order depends on dtype/shape/strides, not numerical entries.
    # Freeze both input allocation branches; exact identity verifies values.
    inverses = []
    for order in ('C', 'F'):
        a = np.array(np.eye(4), order=order)
        b = np.linalg.inv(a.T)
        assert np.array_equal(b, np.eye(4)) and (b.flags.c_contiguous or b.flags.f_contiguous)
        inverses.append(dict(input_order=order, input=layout(a), transposed_input=layout(a.T),
                             output=layout(b), exact_identity=True))
    n = np.empty((4, 4))
    assert n.strides == (32, 8) and n.flags.c_contiguous
    result = dict(complete=True, rays_path=str(RAYS), rays_sha256=RAYS_SHA,
                  npy_header=dict(version=list(version), shape=list(shape), fortran_order=fortran,
                                  dtype=str(dtype)), indexed_rays_layout=indexed,
                  source_sha256=SOURCE_SHA, du_layout=layout(du), dw_layout=layout(dw),
                  cofactor_N_allocation=layout(n), synthetic_inverses=inverses,
                  numpy_version=np.__version__, numpy_inv_source_sha256=hashlib.sha256(inspect.getsource(np.linalg.inv).encode()).hexdigest(),
                  binary_sha256=manifest['binary_sha256'], cpu_seconds=time.process_time()-start,
                  new_synthetic_inverse_calls=2, model_free_duffy_setups=1,
                  new_matrix_products=0, new_mass_model_MC_calls=0,
                  scope='Fixed saved replay layout applicability; generic arbitrary strided rays and original geometry layouts are not inferred.')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
