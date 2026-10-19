import numpy as np, probe
np.set_printoptions(linewidth=220,suppress=True,precision=1)
x=np.random.default_rng(17).standard_normal((200000,10)); h=probe.hidden(x)[:,15]
s=np.sort(x,1)[:,::-1]; am=np.argmax(x,1)
b=np.array([-9,-1.5,-0.5,0.5,1.5,9])
def tab(a,c):
    T=np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            m=(a>=b[i])&(a<b[i+1])&(c>=b[j])&(c<b[j+1]); T[i,j]=h[m].mean() if m.sum()>30 else np.nan
    return T
print('rows x2 bins, cols x5'); print(tab(x[:,2],x[:,5]))
print('rows x2, cols s0-1.5'); print(tab(x[:,2],s[:,0]-1.5))
print('rows x2, cols x1'); print(tab(x[:,2],x[:,1]))
print('rows x2, cols x0'); print(tab(x[:,2],x[:,0]))
print('rows x2-M1 , cols x5'); print(tab(x[:,2]-np.maximum(x[:,0],x[:,1]),x[:,5]))
