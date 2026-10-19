import numpy as np
def feats(x):
    N=x.shape[0]; F={}
    s=np.sort(x,1)[:,::-1]
    am=np.argmax(x,1); o=np.argsort(-x,1); a2=o[:,1]
    F['max']=s[:,0];F['sec']=s[:,1];F['third']=s[:,2];F['gap12']=s[:,0]-s[:,1];F['gap23']=s[:,1]-s[:,2]
    F['am']=am;F['a2']=a2
    for k in range(10): F[f'x{k}']=x[:,k]
    F['x9-max']=x[:,9]-s[:,0]; F['x9-sec']=x[:,9]-s[:,1]
    # running max over 0..8
    m8=x[:,:9].max(1); F['m8']=m8; F['x9-m8']=x[:,9]-m8
    s8=np.sort(x[:,:9],1)[:,::-1]; F['s8_2']=s8[:,1]; F['x9-s8_2']=x[:,9]-s8[:,1]
    return F
