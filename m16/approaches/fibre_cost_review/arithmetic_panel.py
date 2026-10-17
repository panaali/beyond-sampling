"""Same ten frozen cases; native arithmetic independently of historical n+x."""
from pathlib import Path
import json,time
from panel import frozen_models
from arithmetic import evaluate

def main():
    stamp=time.process_time();prior=json.loads(Path(__file__).with_name('panel_results.json').read_text());baselines={r['name']:r for r in prior['rows']};rows=[]
    for name,model,digest in frozen_models():
        start=time.process_time();base=baselines[name];assert digest==base['checkpoint_sha256'];audit=evaluate(model)
        parity={key:audit[key]==base['baseline'][key] for key in ('value_hex','n','x','segments','traces')};assert all(parity.values()),(name,parity)
        # Separate matrix ledger must match the earlier independent named inventory exactly.
        for key in ('mul','add'):assert audit['ledger']['totals']['matrix_'+key]==base['audit']['ledger']['totals'][key]
        named=audit['ledger']['named_numeric_total'];integer=audit['ledger']['integer_control_total']
        row=dict(name=name,checkpoint_sha256=digest,audit=audit,parity=parity,named_numeric_per112=named/112,named_plus_integer_per112=(named+integer)/112,historical_per72=base['historical_sample_equivalents'],cpu_seconds=time.process_time()-start)
        row['named_vs_historical_ratio']=row['named_numeric_per112']/row['historical_per72'];row['named_plus_integer_vs_historical_ratio']=row['named_plus_integer_per112']/row['historical_per72'];rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('audit','parity','checkpoint_sha256')}),flush=True)
    out=dict(rows=rows,cpu_seconds=time.process_time()-stamp,complete_implementation_count=False,scope='Native floating/array arithmetic, comparisons, reductions, sort and special calls. Integer/index/control separate. Memory, selection, dictionary lookup, scalar Boolean control and interpreter overhead excluded. No sampling-MSE verdict.')
    Path(__file__).with_name('arithmetic_panel_results.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
