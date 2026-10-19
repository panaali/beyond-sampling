import numpy as np,sys
d=np.load('d1.npz'); x,h=d['x'][:15000],d['h'][:15000]
p=x[:,:9]; Q=-np.sort(-p,1)
r=lambda a:np.maximum(a,0)
def fsel(X,y):
    m=y>1e-6
    for it in range(6):
        co=np.linalg.lstsq(X[m],y[m],rcond=None)[0]; m=(y>1e-6)|(X@co>0)
    return 1-((np.maximum(X@co,0)-y)**2).mean()/y.var(),co
for arg in sys.argv[1:]:
    i,j=map(int,arg.split(':')); y=h[:,i]
    F={'xj':p[:,j],'Q1':Q[:,0],'Q2':Q[:,1],'Q3':Q[:,2],'Q4':Q[:,3],'x9':x[:,9]}
    o=-np.sort(-np.delete(p,j,1),1); F['O1']=o[:,0];F['O2']=o[:,1];F['O3']=o[:,2]
    if j>0: e=-np.sort(-p[:,:j],1); F['E1']=e[:,0]; 
    if j>1: F['E2']=e[:,1]
    if j<8: l=-np.sort(-p[:,j+1:],1); F['L1']=l[:,0]
    if j<7: F['L2']=l[:,1]
    for k in range(9):
        if k!=j: F[f'x{k}']=p[:,k]
    names=list(F); A=np.stack([F[n] for n in names],1); ch=[]; log=[]
    for st in range(5):
        best=max(((fsel(A[:,ch+[k]],y)[0],k) for k in range(len(names)) if k not in ch))
        ch.append(best[1]); log.append(f'{names[best[1]]}:{best[0]:.3f}')
    print(i,j,' '.join(log), np.round(fsel(A[:,ch],y)[1],2))
