from lib import *
print(12, fit(h[:,12],[relu(m2-x[:,9])]), fit(h[:,12],[relu(S[:,8]-x[:,9])]))
print(16, fit(h[:,16],[relu(x[:,9]-m3)]), fit(h[:,16],[relu(x[:,9]-T[:,8])]),fit(h[:,16],[relu(x[:,9]-S[:,8]),relu(x[:,9]-m3)]))
print(17, fit(h[:,17],[m1,m2,m3]), fit(h[:,17],[m1,m2,m3,T[:,8]]))
print(31, fit(h[:,31],[m1,m2,m3]))
