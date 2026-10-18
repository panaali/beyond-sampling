"""Pin the actually loaded binary and public-source provenance; no cost assumption."""
import contextlib,ctypes,hashlib,inspect,io,json,platform,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent
LIB=Path(np.__file__).parent.parent/'numpy.libs/libscipy_openblas64_-f48b354e.so'
EXT=Path(np.linalg._umath_linalg.__file__)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):np.__config__.show()
    lib=ctypes.CDLL(str(LIB));runtime={}
    for key,symbol in [('config','scipy_openblas_get_config64_'),('core','scipy_openblas_get_corename64_')]:
        f=getattr(lib,symbol);f.restype=ctypes.c_char_p;runtime[key]=f().decode()
    f=lib.scipy_openblas_get_num_threads64_;f.restype=ctypes.c_int;runtime['threads']=f()
    source=inspect.getsource(np.linalg.inv.__wrapped__);(HERE/'installed_python_inv.txt').write_text(source)
    symbols=['scipy_dgesv_64_','dgetrf_single','dgetf2_k','dgetrs_N_single','dtrsm_LNLU','dtrsm_LNUN']
    for symbol in symbols:
        text=subprocess.check_output(['objdump','-d','--disassemble='+symbol,str(LIB)],text=True);(HERE/(symbol+'_disassembly.txt')).write_text(text)
    text=subprocess.check_output(['objdump','-d','--disassemble=_Z3invIdEvPPcPKlS3_Pv',str(EXT)],text=True);(HERE/'numpy_inv_disassembly.txt').write_text(text)
    upstream=[]
    for path,filename in [('lapack/getf2/getf2_k.c','upstream_getf2_k.c'),('lapack/getrf/getrf_single.c','upstream_getrf_single.c'),('lapack/getrs/getrs_single.c','upstream_getrs_single.c'),('driver/level3/trsm_L.c','upstream_trsm_L.c')]:
        p=HERE/filename;upstream.append(dict(url='https://raw.githubusercontent.com/OpenMathLib/OpenBLAS/v0.3.34/'+path,download_date='2026-10-17',local=filename,sha256=digest(p),exact_build_source_equivalence_not_yet_proved=True))
    out=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),python_inv_file=inspect.getsourcefile(np.linalg.inv.__wrapped__),binary_paths={'numpy_linalg':str(EXT),'openblas':str(LIB)},binary_hashes={'numpy_linalg':digest(EXT),'openblas':digest(LIB)},runtime=runtime,build_config=buf.getvalue(),public_upstream=upstream,ldd=subprocess.check_output(['ldd',str(EXT)],text=True))
    (HERE/'environment.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('build_config','public_upstream','ldd')},indent=2))
if __name__=='__main__':main()
