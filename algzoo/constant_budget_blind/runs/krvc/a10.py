from fitlib import *
ev(7,['r(x9-M9)']); ev(12,['r(S9-x9)']); ev(16,['r(x9-S9)']); ev(16,['x9-S9'])
ev(17,['M9','S9'],relu=False); ev(31,['S9'],relu=False); ev(31,['S9','M9'],relu=False)
ev(1,['g12','am8*g12']); ev(1,['g12','am8*g12','am1*g12'])
ev(9,['S9','T9']); ev(9,['S9','T9','x9'])
ev(5,['g12','am8*g12','am2*g12','am5*g12'])
ev(21,['x9-M9','r(x9-S9)']); ev(23,['x9-M9','r(x9-S9)','am8']); ev(23,['r(x9-S9)','am8*g12','am8'])
