from lib import *
np.set_printoptions(precision=2,suppress=True,linewidth=160)
n=50000; x=x[:n]; M=M[:n]; S=S[:n]
h0=np.load('h0.npy'); hn=np.load('hn.npy')
w=np.array([0,0,-0.2103,-0.1674,-0.2445,-1.2097,0.1348,1.3719])
g=h0.copy()
for i in [2,3,4,5]:
    m=h0[:,i]<=0; g[m,i]=hn[m,i]-10*(-w[i])
np.save('g.npy',g)
print('frac g<=0 remaining',(g<=0).mean(0))
sx=-np.sort(-x[:,:9],1)
def fit(F,y):
    F=np.c_[F,np.ones(len(y))]; c,*_=np.linalg.lstsq(F,y,rcond=None); r=y-F@c; return 1-r.var()/y.var(),c
for i in range(8):
    y=g[:,i]; ok=y>0
    a=fit(sx[ok],y[ok]); b=fit(np.c_[sx,x[:,:9]][ok],y[ok]); 
    print(i,'sorted',round(a[0],3),a[1][:9],' +x',round(b[0],3)); print('    x part',b[1][9:18])
