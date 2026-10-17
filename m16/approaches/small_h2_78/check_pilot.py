import json,time
from pathlib import Path
import numpy as np
from envelope import envelope,reset_certificate,nonvanishing_certificate


def main():
    start=time.process_time();p=Path(__file__).with_name('LOCALPILOT_H2_T7_seed0.npz');data=np.load(p)
    base=tuple(data[k].astype(float) for k in ('Wi','Wh','Wo'));rows=[]
    for sigma in (0.,.0025,.02):
        rng=np.random.default_rng(1)
        model=tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.standard_normal(w.shape) for w in base)
        cert=nonvanishing_certificate(model)
        assert cert['holds']
        for scale in (1e-16,1e16):
            alt=nonvanishing_certificate((model[0]*scale,model[1],model[2]*scale))
            assert alt['holds'] and alt['nonzero_state_output_labels']==cert['nonzero_state_output_labels']
        rows.append({'sigma':sigma,'nonvanishing_certificate':cert,'two_negative_reset':reset_certificate(model)})
    out={'rows':rows,'cpu_seconds':time.process_time()-start,'model_input_evaluations':0,
         'rational_certificates':9,'classification':'structural certificate, not accuracy estimator'}
    Path(__file__).with_name('pilot_certificates.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
