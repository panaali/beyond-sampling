import json
M,S,T="topk(x,9,0)","topk(x,9,1)","topk(x,9,2)"
sp={}
for i,j in [(0,2),(4,4),(13,5),(7,6),(27,0)]:
    sp[i]=['relu',['x[:,%d]'%j,'so(x,%d)'%j,'I1(x,%d)'%j,'I2(x,%d)'%j],None]
sp[25]=['relu',['rSo9(x,1)','rSo9(x,3)','I1(x,1)','I2(x,1)'],None]
sp[29]=['relu',['rSo9(x,1)','topk(x,4,0)','x[:,1]','rSo9(x,0)'],None]
sp[28]=['gated',['x[:,3]','So9(x,3)','am9(x)','Mo9(x,3)'],'1-top2(x,3)']
sp[24]=['gated',['x[:,7]','So9(x,7)','so(x,7)'],'1-top2(x,7)']
sp[3]=['gated',['x[:,8]','So9(x,8)','am9(x)'],'1-top2(x,8)']
sp[10]=['gated',['x[:,8]','So9(x,8)','am9(x)','Mo9(x,8)'],'1-top2(x,8)']
for i,s in [(1,[0,2,3]),(2,[8]),(8,[0,2,4,6,8]),(15,[1,4]),(20,[0,1,2,3,4,5,6,7]),(18,[0,1,2,3,4,5,6]),(19,[2,3,5,6,8])]:
    sp[i]=['relu',[M,S,'x[:,9]','inset(am9(x),%s)'%s],None]
sp[5]=['relu',[M,S,'inset(am9(x),[1,5,7])'],None]
sp[12]=['relu',[M,S,'inset(am9(x),[0,2,3,4,6])'],None]
sp[14]=['gated',[M,S,'I1(x,3)'],'inset(am9(x),[0,1,2,3,5])']
sp[6]=['relu',[S,M,'x[:,9]'],None]
sp[9]=['scale',['relu(x[:,8]-topk(x,8,0))'],None]
sp[22]=['scale',['relu(x[:,9]-topk(x,9,0))'],None]
sp[16]=['gated',['x[:,4]','so(x,4)'],'I1(x,4)']
sp[23]=['gated',['x[:,7]','so(x,7)'],'I1(x,7)']
sp[26]=['relu',['topk(x,8,0)',S],None]
sp[31]=['relu',['x[:,9]',S],None]
sp[21]=['lin',['x[:,:8].sum(1)',M,S,T],None]
sp[17]=['relu',['rSo9(x,8)','J1(x,6)','I2(x,7)','topk(x,5,0)'],None]
sp[11]=['relu',['rSo9(x,8)','I1(x,2)','rMo9(x,8)','topk(x,6,0)'],None]
sp[30]=['gated',['x[:,8]','So9(x,8)','x[:,9]'],'I1(x,8)']
assert sorted(sp)==list(range(32))
json.dump({str(k):sp[k] for k in sorted(sp)},open('spec.json','w'),indent=0)
sp[0]=['relu',['x[:,2]','I1(x,2)','I2(x,2)','inset(am9(x),[0])'],None]
sp[4]=['relu',['x[:,4]','I1(x,4)','I2(x,4)','inset(am9(x),[1,2])'],None]
sp[13]=['relu',['x[:,5]','I1(x,5)','I2(x,5)','rSo9(x,5)'],None]
sp[30]=['gated',['J1(x,8)','x[:,9]','s_all(x,1)-s_all(x,2)'],'pair30(x)']
json.dump({str(k):sp[k] for k in sorted(sp)},open('spec.json','w'),indent=0)
sp[3]=['gated',['x[:,8]',M,S,'inset(am9(x),[5,6])'],'1-top2(x,8)']
sp[28]=['gated',['x[:,3]',M,S,'inset(am9(x),[1,5,6,7,8])'],'1-top2(x,3)']
json.dump({str(k):sp[k] for k in sorted(sp)},open('spec.json','w'),indent=0)
