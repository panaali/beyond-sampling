import numpy as np, probe
np.set_printoptions(linewidth=220)
X=[]
for p in range(10):
    for q in range(10):
        x=np.zeros(10); x[q]=2.0; x[p]=3.0  # p==q -> single spike 3
        X.append(x)
X=np.array(X); H=probe.hidden(X).reshape(10,10,16)
print('pred',probe.predict(X).reshape(10,10))
for i in range(16):
    print('neuron',i,' rows=p(max=3), cols=q(second=2); diag=single spike')
    print(np.round(H[:,:,i]).astype(int))
