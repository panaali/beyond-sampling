"""Frozen12 native parity and expression costs; no result-based tuning."""
from pathlib import Path
import json,time
from loader import load
from models import frozen_models
from sampler_native import count

def evaluate(model,instrument):
    mod,_,ledger=load(instrument);assert mod.LOOK==2;assert all(getattr(mod,k) for k in ('PRUNE','RANGEF','INCR','CLAMPC','FLIP','SHARE'))
    ops=mod.Ops();info={};value=mod.est_fast(model,ops,info)
    out=dict(value=value,value_hex=float(value).hex(),n=ops.n,x=getattr(ops,'x',0),traces=info['traces'],segments=info['segments'],cons=info['cons'],err_by={str(k):float(v).hex() for k,v in info['err_by'].items()})
    if instrument:out['ledger']=ledger.report()
    return out

def main():
    start=time.process_time();rows=[]
    for name,model,digest in frozen_models():
        stamp=time.process_time();base=evaluate(model,False);audit=evaluate(model,True);parity={k:base[k]==audit[k] for k in base};assert all(parity.values()),(name,parity)
        if name in ('zero','constant_zero_readout','constant_duplicate_readout'):assert abs(audit['value']-1/3)<1e-14,(name,audit['value'])
        sample=count(model);sampletotal=sample['named_numeric_total']+sample['integer_control_total'];assert sampletotal==112
        total=audit['ledger']['named_numeric_total']+audit['ledger']['integer_control_total'];historical=audit['n']+audit['x'];row=dict(name=name,checkpoint_sha256=digest,baseline=base,audit=audit,parity=parity,sampler=sample,expression_total=total,historical_nx=historical,expression_sample_equivalents=total/112,historical_nx_per72=historical/72,cpu_seconds=time.process_time()-stamp);row['budget_ratio_vs_historical_nx']=row['expression_sample_equivalents']/row['historical_nx_per72'];row['historical_reported_n_per72']=audit['n']/72;row['budget_ratio_vs_reported_n']=row['expression_sample_equivalents']/row['historical_reported_n_per72'];rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('baseline','audit','parity','sampler','checkpoint_sha256')}),flush=True)
        Path(__file__).with_name('panel_results.json').write_text(json.dumps(dict(rows=rows,cpu_seconds=time.process_time()-start,completed=len(rows),expected=12,scope='explicit source arithmetic only; same source-form/control/memory exclusions as T121; no MSE verdict'),indent=2)+'\n')
if __name__=='__main__':main()
