import numpy as np

def predict(x,payload):
    target,version,wi,wh,wo=payload
    if version!=1 or target not in ('hidden','labels'):raise ValueError('header')
    wi=np.asarray(wi,dtype=np.float64);wh=np.asarray(wh,dtype=np.float64)
    h=np.zeros((len(x),16),dtype=np.float64)
    for j in range(10):
        recurrent=h.dot(wh.T);forced=np.multiply(x[:,j,None],wi[None,:]);h=np.maximum(np.add(recurrent,forced),0.)
    if target=='hidden':
        if wo is not None:raise ValueError('conditional readout')
        return h
    return h.dot(np.asarray(wo,dtype=np.float64).T).argmax(axis=1)
