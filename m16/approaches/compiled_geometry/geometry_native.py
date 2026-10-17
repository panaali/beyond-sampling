"""T3-only native cone lifecycle; returns copied arrays before freeing C++ state."""
import ctypes,time
from pathlib import Path
import numpy as np
LIB=ctypes.CDLL(str(Path(__file__).parent/'data/dd_geometry.so'))
P=ctypes.c_void_p;I=ctypes.c_int;L=ctypes.c_longlong
LIB.dd_geometry.argtypes=[I,P,P,P,P,P];LIB.dd_geometry.restype=P
LIB.dd_get.argtypes=[P,I,P];LIB.dd_get.restype=P
LIB.dd_free.argtypes=[P];LIB.dd_free.restype=None
LIB.dd_rank_test.argtypes=[I,P];LIB.dd_rank_test.restype=I

def geometry(model):
    wi,wh,wo=(np.ascontiguousarray(x,dtype=np.float64) for x in model);h=len(wi)
    if wi.shape!=(h,) or wh.shape!=(h,h) or wo.shape!=(3,h) or not 1<=h<=16:raise ValueError('H<=16,T3 only')
    if not all(np.isfinite(x).all() for x in (wi,wh,wo)):raise ValueError('finite weights required')
    order=np.asarray(np.argsort(-abs(wi)),dtype=np.int32)
    code=I();start=time.process_time();wall=time.perf_counter();ptr=LIB.dd_geometry(h,wi.ctypes.data,wh.ctypes.data,wo.ctypes.data,order.ctypes.data,ctypes.addressof(code));kernel=time.process_time()-start;kernelwall=time.perf_counter()-wall
    if code.value:raise RuntimeError(f'native DD lifecycle status {code.value}')
    start=time.process_time();wall=time.perf_counter();arrays={}
    try:
        for tag,name,dtype,ctype,shape in [(0,'good',np.float64,ctypes.c_double,(-1,3,3)),(1,'bad',np.float64,ctypes.c_double,(-1,3,3)),(2,'rays',np.float64,ctypes.c_double,(-1,3)),(3,'bits',np.uint64,ctypes.c_uint64,(-1,2)),(4,'headers',np.int32,ctypes.c_int,(-1,2)),(5,'maps',np.float64,ctypes.c_double,(-1,h,3)),(6,'counts',np.int64,L,(-1,))]:
            size=L();raw=LIB.dd_get(ptr,tag,ctypes.addressof(size));arr=np.ctypeslib.as_array(ctypes.cast(raw,ctypes.POINTER(ctype)),shape=(size.value,)).copy() if size.value else np.empty(0,dtype=dtype);arrays[name]=arr.reshape(shape)
    finally:LIB.dd_free(ptr)
    if not all(np.isfinite(arrays[key]).all() for key in ('good','bad','rays','maps')):raise FloatingPointError('nonfinite native geometry output')
    arrays['timing']={'kernel_cpu':kernel,'kernel_wall':kernelwall,'serialization_cpu':time.process_time()-start,'serialization_wall':time.perf_counter()-wall}
    return arrays

def rank(matrix):
    matrix=np.ascontiguousarray(matrix,dtype=np.float64)
    if matrix.ndim!=2 or matrix.shape[1]!=3:raise ValueError('3 columns required')
    return LIB.dd_rank_test(len(matrix),matrix.ctypes.data)
