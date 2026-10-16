import json,numpy as np
A="data/acts"
shape=tuple(json.load(open(f"{A}/meta.json"))["shape"]); n=json.load(open(f"{A}/progress.json"))["done"]
acts=np.memmap(f"{A}/acts.f16",dtype=np.float16,mode="r",shape=shape)
print(shape,n)
mx=[];infs=0
for L in range(shape[2]):
    b=np.asarray(acts[:n,:,L,:])
    infs+=np.isinf(b).sum()
    mx.append(float(np.abs(b[:,4:].astype(np.float32)).max()))
    if L in (0,5,14,27):
        # pos 0-3 vs rest magnitude
        print(L,"absmax pos0-3",np.abs(b[:,:4].astype(np.float32)).max(), "rest", mx[-1])
print("infs",infs,"max",max(mx))
# tiny values: fp16 resolution near zero for neurons
b=np.asarray(acts[:n,4:,10,:32]).astype(np.float32)
print("frac exactly zero neuron L10",(b==0).mean())
