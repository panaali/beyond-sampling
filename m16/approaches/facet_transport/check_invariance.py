"""Post-review regression: positive Wi/Wo scaling on the same frozen cases."""
import json,time
from pathlib import Path
import probe as P
import numpy as np
import torch

def main():
    cpu=time.process_time();wall=time.monotonic()
    r=np.random.default_rng(31001)
    synthetic=(r.standard_normal(3),r.standard_normal((3,3)),r.standard_normal((3,3)))
    state=torch.load('/workspace/work/checkpoints/H4_T3_seed0.pth',map_location='cpu',weights_only=True)
    w=[state[k].numpy().astype(float) for k in ['rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight']]
    base=(w[0][:,0],w[1],w[2]);cases=[('synthetic',synthetic)]
    for sigma in [.0025,.02]:
        r=np.random.default_rng(1)
        cases.append((f'dev_H4_T3_s{sigma}',tuple(w+sigma*np.sqrt(np.mean(w*w))*r.standard_normal(w.shape) for w in base)))
    original=json.loads(Path(__file__).with_name('results.json').read_text())
    report={'cases':[],'scale':1e-16,'note':'Same frozen models, deterministic geometry only; no new candidate or sampled inputs.'}
    for (name,model),old in zip(cases,original['cases']):
        assert name==old['name']
        wi,wh,wo=model
        delta=np.random.default_rng(31002).standard_normal(wo.shape)*.1*np.sqrt(np.mean(wo*wo))
        row={'name':name,'variants':[]}
        for label,scaled,delta_scaled in [('nominal',model,delta),('Wi_times_1e-16',(wi*1e-16,wh,wo),delta),('Wo_times_1e-16',(wi,wh,wo*1e-16),delta*1e-16)]:
            gates,stages=P.gate_complex(scaled);ranked=P.rank_complex(gates)
            acc=P.accuracy(ranked,scaled[2]);flux,geometry=P.flux(ranked,scaled[2],delta_scaled)
            counts_same=(len(gates)==old['gate_cells'] and len(ranked)==old['rank_cells'] and
                         all(geometry[k]==old[k] for k in ['decision_arcs','unique_planes','merged_arcs']))
            record=dict(variant=label,accuracy=acc,flux=flux,accuracy_difference=acc-old['accuracy'],
                flux_difference=flux-old['flux'],gate_cells=len(gates),rank_cells=len(ranked),
                original_candidate_counts_unchanged=counts_same,**geometry)
            assert counts_same,record
            assert abs(record['accuracy_difference'])<1e-11,record
            assert abs(record['flux_difference'])<1e-10,record
            row['variants'].append(record)
        report['cases'].append(row)
    report.update(cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall,counts=P.COUNTS)
    Path(__file__).with_name('invariance_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
