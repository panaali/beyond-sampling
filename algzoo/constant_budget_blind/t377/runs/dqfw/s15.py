from lib import *
x9=x[:,9]; M7=rm[:,7]; M8=rm[:,8]; S7=S[:,7]; S8=S[:,8]
print(1, fit(h[:,1],[relu(M7-S8)]), fit(h[:,1],[relu(M7-S8),x9]))
print(17, fit(h[:,17],[M8,S8]), fit(h[:,17],[M8,S8,T[:,8]]), fit(h[:,17],[m1,m2,m3,M8]))
print(31, fit(h[:,31],[S8]), fit(h[:,31],[S8,M8]), fit(h[:,31],[S8,M8,T[:,8]]))
nr8=(R[:,8]==0)
print(23, fit(h[:,23],[relu(x9-m3)*nr8]), relufit(h[:,23],np.stack([x9*nr8,m3*nr8],1)), relufit(h[:,23],np.stack([x9,m3,R[:,8]],1)))
print(21, relufit(h[:,21],np.stack([x9,m3,R[:,9],R[:,7],M7],1)), fit(h[:,21],[relu(x9-m3)]), fit(h[:,21],[relu(x9-m3)*(R[:,7]==0)]),fit(h[:,21],[relu(x9-m3)*(R[:,7]==0)*nr8]))
