from lib import *
x9=x[:,9]; M8=rm[:,8]; S8=S[:,8]
for i,P,gate8 in [(0,[1,2,5,6],1),(13,[1,2,4,7],1),(29,[0,3,6,7],1),(6,[0,2,5,7],0),(18,[0,1,4,6,8],0)]:
    y=h[:,i]; inP=np.isin(am,P)*1.; g=(am!=8)*1. if gate8 else np.ones(N)
    b=relu(S8-x9)*g
    c1,q1=fit(y,[b,b*inP])
    A=np.stack([S8*g,M8*g,x9*g,S8*g*inP,M8*g*inP],1); c2,q2=relufit(y,A)
    A=np.stack([S8*g,M8*g,x9*g,S8*g*inP],1); c3,q3=relufit(y,A)
    print(i,P,'T1 %.3f'%q1,np.round(c1,2),'T2 %.3f'%q2,np.round(c2,2),'T3 %.3f'%q3,np.round(c3,2))
