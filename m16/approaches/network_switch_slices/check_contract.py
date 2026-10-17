import hashlib,json,time
from .slices import ROOT
from .check_slices import HERE
from .contract import contract


def main():
    start=time.process_time()
    import torch
    assert json.loads((HERE/'endpoint_results.json').read_text())['passed']
    prior=json.loads((ROOT/'m16/approaches/tensor_negative_review/grid_results.json').read_text())
    truth={2:.30773074302315223,4:.47701312599404416}
    out=dict(cases=[],sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [HERE/'slices.py',HERE/'contract.py']})
    def save():
        out['cpu_seconds']=time.process_time()-start
        (HERE/'contraction_results.json').write_text(json.dumps(out,indent=2)+'\n')
    for H in (2,4):
        checkpoint=ROOT/f'work/checkpoints/H{H}_T4_seed0.pth'
        st=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model=(st['rnn.weight_ih_l0'].numpy().astype(float).ravel(),st['rnn.weight_hh_l0'].numpy().astype(float),st['linear.weight'].numpy().astype(float))
        for order in (3,5):
            try: row=contract(model,order,16,32,deadline=start+105)
            except Exception as error:
                out['failure']=dict(H=H,outer_order=order,error=str(error),
                                    total_cpu_at_failure=time.process_time()-start)
                save();raise
            old=next(r for r in prior['cases'] if r['H']==H and r['order']==order)
            row.update(H=H,T=4,reference=truth[H],error_to_reference=row['estimate']-truth[H],
                       old_full_grid_estimate=old['full_grid_accuracy'],old_full_grid_bias=old['full_grid_bias'],
                       checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            out['cases'].append(row);save()
            print(json.dumps({k:row[k] for k in ('H','outer_order','estimate','diagnostic_estimate','error_to_reference','old_full_grid_bias','total_cpu_seconds')}),flush=True)
    out['complete']=True;save()


if __name__=='__main__':main()
