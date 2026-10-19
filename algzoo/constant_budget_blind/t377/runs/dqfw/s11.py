from lib import *
for i,p in [(22,7),(19,6),(2,3),(15,1),(26,2),(14,8),(8,5),(10,4),(10,5),(11,4),(27,2),(27,6),(3,6)]:
    y=h[:,i]; xp=x[:,p]; xo=np.delete(x,p,1); so=-np.sort(-xo,1)
    t2=((am==p)|(a2==p)).astype(float); g=(am==p).astype(float)
    F={'r(xp-m2o)':relu(xp-so[:,1]),'r(xp-mo)':relu(xp-so[:,0]),'t2*xp':t2*xp,'g*xp':g*xp,'t2':t2,'g':g,'t2*mo':t2*so[:,0],'t2*m3':t2*m3,'g*m2':g*m2}
    res=[]
    for combo in [['r(xp-m2o)'],['r(xp-m2o)','r(xp-mo)'],['t2*xp','g*xp'],['t2*xp','g*xp','t2*m3'],['r(xp-m2o)','r(xp-mo)','t2*xp'],['t2*xp','g*xp','t2*m3','g*m2']]:
        c,r=fit(y,[F[k] for k in combo]); res.append('%s:%.3f'%('+'.join(combo),r))
    print(i,p,' '.join(res))
