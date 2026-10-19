import numpy as np, feats, probe
np.set_printoptions(precision=2,suppress=True,linewidth=230)
rng=np.random.default_rng(13)
x=rng.standard_normal((40000,10)); H=probe.hidden(x)
F,u9=feats.build2(x); names=list(F)
pre=x[:,:9]; am=pre.argmax(1); a2=np.argsort(-pre,1)[:,1]
def sub(m,i,lab):
    h=H[m,i]; Fs={n:F[n][m] for n in names}
    hist=feats.greedy(h,Fs,names,3)
    print(i,lab,'n=%d mean %.2f act %.2f'%(m.sum(),h.mean(),(h>0).mean()),' | '.join('%.3f %s'%(r,s[-1]) for r,s,c in hist),hist[-1][2])
sub(am==8,10,'pam=8')
print('10 mean by a2 when pam=8',[round(H[(am==8)&(a2==k),10].mean(),2) for k in range(8)])
sub(am!=8,10,'pam!=8')
rk8=(pre>x[:,[8]]).sum(1)
sub(rk8>=2,31,'x8 rank>=2 in prefix')
print('31 mean by pam when x8 rank>=2',[round(H[(rk8>=2)&(am==k),31].mean(),2) for k in range(8)])
print('31 mean by pa2 when x8 rank>=2',[round(H[(rk8>=2)&(a2==k),31].mean(),2) for k in range(8)])
