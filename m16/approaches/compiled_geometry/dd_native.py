"""Local ctypes adapter; original tolerance-based DD semantics retained."""
import ctypes
from pathlib import Path
import numpy as np
LIB=ctypes.CDLL(str(Path(__file__).parent/'data/dd_clip.so'))
P=ctypes.c_void_p;I=ctypes.c_int
LIB.dd_clip.argtypes=[I,I,P,P,P,I,I,I,P,P,P,P];LIB.dd_clip.restype=I
LIB.dd_replay.argtypes=[I,P,P,P,P,P,P,P];LIB.dd_replay.restype=I

def status(code):
    if code==1:raise ValueError('clip: zero normal')
    if code==2:raise FloatingPointError('nonfinite/degenerate clip arithmetic')
    if code:raise RuntimeError(f'native DD status {code}')

def clip(R,Z,a,cid,ops,want=(True,True)):
    R=np.ascontiguousarray(R,dtype=np.float64);Z=np.ascontiguousarray(Z,dtype=np.uint64);a=np.ascontiguousarray(a,dtype=np.float64)
    k,d=R.shape
    if Z.shape!=(k,2) or a.shape!=(d,) or not 0<=cid<128:raise ValueError('DD shape/constraint id')
    capacity=k+k*k//4
    rays=np.empty((2,capacity,d));bits=np.empty((2,capacity,2),dtype=np.uint64);sizes=np.empty(2,dtype=np.int32);cost=np.empty(2,dtype=np.int64)
    status(LIB.dd_clip(k,d,R.ctypes.data,Z.ctypes.data,a.ctypes.data,cid,int(want[0])+2*int(want[1]),capacity,rays.ctypes.data,bits.ctypes.data,sizes.ctypes.data,cost.ctypes.data))
    ops.add(int(cost[0]),'clip');ops.add(int(cost[1]),'adj')
    return tuple(None if sizes[s]<0 else (rays[s,:sizes[s]].copy(),bits[s,:sizes[s]].copy()) for s in range(2))

def pack(corpus):
    meta=[];offsets=[];rays=[];bits=[];normals=[];nr=nz=na=0
    for R,Z,a,cid,want in corpus:
        meta.append((len(R),R.shape[1],cid,int(want[0])+2*int(want[1])));offsets.append((nr,nz,na));rays.append(R.ravel());bits.append(Z.ravel());normals.append(a);nr+=R.size;nz+=Z.size;na+=a.size
    return np.asarray(meta,dtype=np.int32),np.asarray(offsets,dtype=np.int64),np.concatenate(rays),np.concatenate(bits),np.concatenate(normals)

def replay(packed):
    meta,off,R,Z,a=packed;answer=np.zeros(len(meta));totals=np.zeros(2,dtype=np.int64)
    status(LIB.dd_replay(len(meta),meta.ctypes.data,off.ctypes.data,R.ctypes.data,Z.ctypes.data,a.ctypes.data,answer.ctypes.data,totals.ctypes.data))
    return answer,totals
