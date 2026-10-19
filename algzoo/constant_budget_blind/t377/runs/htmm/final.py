import numpy as np,json
from lib import fit_relu
from fe import base,feat
S={0:'d2,AM02,T22,ABS0123456',1:'x9,Q8,AM03,AM02',2:'x9,P8,AM8,AM24',3:'q8,AM56,AM37,A258',4:'T24,AM12,d4,ABS0123456',
5:'P8,Q8,AM17,AM35',6:'x9,Q8',7:'d6,T26,ABS01234567,Q8',8:'x9,Q8,AM46,AM08',9:'x8,P7',10:'T28,Q8,T267,ABS0123456',
11:'AM248,A28,AM3467,P7',12:'P8,Q8,AM0',13:'T25,x5,ABS0123456,R8',14:'AM4678,P8,AM378,AM1578',15:'x9,Q8,AM14,AM13',
16:'AM4,d4,ABS01234567,A235',17:'AM1246,A268,AM0236,Q8',18:'x9,P8,Q8,AM78',19:'x9,P8,Q8',20:'x9,P8,AM8,AM2',
21:'ABS0123456,R8,T245,q2',22:'x9,P8',23:'AM7,q7,ABS01234567,q8',24:'q7,A267,Q8,T27',25:'AM13,T21,P8,T213',26:'G26',
27:'T20,x0,T212,ABS0123456',28:'q3,T23,ABS01234567,AM02',29:'q1,AM03,P7,T21',30:'AM8,A23567,x9,Q8',31:'x9,Q8'}
xa=np.load('x2.npy'); ha=np.load('h2.npy'); xb=np.load('x3.npy'); hb=np.load('h3.npy')
Da=base(xa); Db=base(xb); out={}
for i in range(32):
    fs=S[i].split(',')
    b,r=fit_relu(np.stack([feat(Da,f) for f in fs],1),ha[:,i],it=30)
    pr=np.maximum(np.c_[np.stack([feat(Db,f) for f in fs],1),np.ones(len(xb))]@b,0)
    r2=1-((pr-hb[:,i])**2).mean()/hb[:,i].var()
    out[i]=dict(f=fs,b=[round(float(v),4) for v in b],r2=round(float(r2),4))
    print(i,fs,out[i]['b'],'test R2 %.4f'%r2)
json.dump(out,open('spec.json','w'))
