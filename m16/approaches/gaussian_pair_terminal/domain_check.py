import json,warnings
from pathlib import Path
import numpy as np
from terminal import PairTerminal
HERE=Path(__file__).resolve().parent
rows=[]
for name,model,hidden,limit,u in (
 ('underflow_rank_mass',(np.ones(2),np.eye(2),np.zeros((4,2))),np.zeros(2),40.,39.),
 ('overflow_hidden_map',(np.ones(2),np.full((2,2),1e308),np.zeros((4,2))),np.ones(2),6.,0.)):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            PairTerminal(model,hidden,0,limit=limit).probabilities(u)
            raise AssertionError('domain failure not rejected')
        except ArithmeticError as exc:rows.append(dict(name=name,rejected=str(exc),warnings=[str(w.message) for w in caught]))
(HERE/'domain_check.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows))
