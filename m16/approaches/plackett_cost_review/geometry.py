"""Unchanged geometry with mass callback replaced by exact ordered-ray validation."""
import argparse,json,resource,time
import numpy as np
from inventory import HERE,AUTHOR,Ledger,Ops,load


def execute(model,first,last,instrument,expected=None):
    E,ledger,startup=load(instrument);halves=E.halves(4);setup=ledger.report()
    ledger=Ledger();E._arithmetic=ledger;E._HALVES[4]=halves[first:last]
    count=0;rays=[]
    def record(R,o):
        nonlocal count
        if expected is not None:assert np.array_equal(R,expected[count]),('ordered ray mismatch',first,last,count)
        else:rays.append(np.array(R,copy=True))
        count+=1;E.STATS['simplices']+=1;return 0.
    E.orthant=record;E.STATS.update({k:0 for k in E.STATS});ops=Ops();value=E.estT(model,ops,mode='exact')
    assert value==1.
    if expected is not None:assert count==len(expected)
    return dict(first=first,last=last,native=ledger.report(),cold_setup=setup,rule_setup=startup,legacy_geometry=ops.n,stats=dict(E.STATS),mass_callbacks=count),rays


def controls():
    start=time.process_time();model=(np.zeros(2),np.zeros((2,2)),np.zeros((4,2)))
    baseline,rays=execute(model,0,96,False);traced,again=execute(model,0,96,True,np.asarray(rays))
    for key in ('legacy_geometry','stats','mass_callbacks'):assert baseline[key]==traced[key],key
    assert traced['stats']['cells']==96
    traced.update(cpu_seconds=time.process_time()-start,baseline=baseline)
    (HERE/'geometry_controls.json').write_text(json.dumps(traced,indent=2)+'\n');print(json.dumps({k:v for k,v in traced.items() if k not in ('native','cold_setup','rule_setup','baseline')}))


def batch(index):
    start=time.process_time();assert 0<=index<8
    import torch
    torch.set_num_threads(1);s=torch.load('/workspace/work/t45_weights/2nd_argmax_8_4_1073741824_0.pth',map_location='cpu',weights_only=True)
    model=(s['rnn.weight_ih_l0'].numpy().astype(float)[:,0],s['rnn.weight_hh_l0'].numpy().astype(float),s['linear.weight'].numpy().astype(float))
    archive=np.load(AUTHOR/'leaf_mass_review/error_captured.npz');parents=archive['parent_ids'];selected=(parents>=12*index)&(parents<12*(index+1));expected=archive['rays'][selected]
    result,_=execute(model,12*index,12*(index+1),True,expected)
    result.update(batch=index,cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (HERE/f'geometry_{index}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('native','cold_setup','rule_setup')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase');a=p.parse_args()
    try:controls() if a.phase=='controls' else batch(int(a.phase))
    except Exception as exc:
        (HERE/f'geometry_{a.phase}_failure.json').write_text(json.dumps(dict(failure=repr(exc)),indent=2)+'\n');raise
