def predict(x,payload):
    target,version,wo=payload
    if version!=1 or target not in ('hidden','labels'):raise ValueError('header')
    h=np.stack([f(i,x) for i in range(16)],axis=1)
    if target=='hidden':
        if wo is not None:raise ValueError('conditional readout')
        return h
    return (h@np.asarray(wo,dtype=np.float64).T).argmax(axis=1)
