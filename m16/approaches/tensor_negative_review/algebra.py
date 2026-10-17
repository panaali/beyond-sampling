"""Independent full-tensor algebra references, not a TT-cross implementation."""
import numpy as np

def product_weights(weights):
    result=np.array(1.)
    for w in weights:result=np.multiply.outer(result,w)
    return result

def direct_integral(tensor,weights):
    return float(np.sum(tensor*product_weights(weights)))

def reconstruct(cores):
    result=cores[0]
    for core in cores[1:]:result=np.tensordot(result,core,axes=(-1,0))
    return result.reshape(tuple(core.shape[1] for core in cores))

def contract(cores,weights):
    """Explicit scalar state recursion, independent of author's einsum."""
    state=[1.]
    for core,w in zip(cores,weights):
        following=[0.]*core.shape[2]
        for left in range(core.shape[0]):
            for node in range(core.shape[1]):
                for right in range(core.shape[2]):
                    following[right]+=state[left]*float(w[node])*float(core[left,node,right])
        state=following
    assert len(state)==1
    return state[0]

def weighted_svd(tensor,weights,rank):
    """Full-data weighted TT-SVD; oracle diagnostic, never a cheap estimator."""
    scaled=tensor*np.sqrt(product_weights(weights));shape=tensor.shape
    residual=scaled;left=1;cores=[];discarded=0.;kernels=[]
    for axis,n in enumerate(shape[:-1]):
        matrix=residual.reshape(left*n,-1)
        u,s,vt=np.linalg.svd(matrix,full_matrices=False)
        keep=min(rank,len(s));discarded+=float(s[keep:]@s[keep:])
        kernels.append(dict(shape=list(matrix.shape),retained=keep,singular_values=s.tolist()))
        cores.append(u[:,:keep].reshape(left,n,keep)/np.sqrt(weights[axis])[None,:,None])
        residual=s[:keep,None]*vt[:keep];left=keep
    cores.append(residual.reshape(left,shape[-1],1)/np.sqrt(weights[-1])[None,:,None])
    return cores,dict(weighted_l2_upper_bound=float(np.sqrt(discarded)),svd_kernels=kernels)
