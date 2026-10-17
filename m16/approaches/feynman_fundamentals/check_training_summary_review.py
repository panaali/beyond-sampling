"""Independent Decimal training-panel ratios from raw rows, no author imports."""
import decimal
import hashlib
import json
import time
from pathlib import Path

start=time.process_time();decimal.getcontext().prec=50;D=decimal.Decimal
base=Path('/workspace/work/recover-small_h2_56/m16/approaches/training_axis_ladder')
saved=json.loads((base/'summary.json').read_text())
out=[];hashes={};models=set()
for seed in range(5):
    rows=[];panels=[]
    for noise in (20,21,22,23):
        path=base/f'block_s{seed}_n{noise}.json'
        block=json.loads(path.read_text(),parse_float=D)
        assert block['complete'] and len(block['rows'])==256
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        rows.extend(block['rows'])
        for row in block['rows']:
            assert row[-1] not in models;models.add(row[-1])
    n=D(len(rows));mse=sum((r[2]-r[1])**2 for r in rows)/n
    meanops=sum(D(r[3]) for r in rows)/n
    pq=sum(r[1]*(1-r[1]) for r in rows)/n
    matched=sum(D(72)*r[1]*(1-r[1])/D(r[3]) for r in rows)/n
    integer=sum(r[1]*(1-r[1])/D(r[3]//72) for r in rows)/n
    values=dict(mse_ratio=float(mse*meanops/(D(72)*pq)),model_matched_ratio=float(mse/matched),
                integer_budget_matched_ratio=float(mse/integer))
    ref=next(r for r in saved['seeds'] if r['training_seed']==seed)
    out.append(dict(training_seed=seed,**values,
                    maximum_relative_difference=max(abs(v/ref[k]-1) for k,v in values.items()),
                    single_copy_ratios_above_one=sum((r[2]-r[1])**2*D(r[3])>D(72)*r[1]*(1-r[1]) for r in rows)))
raw=json.loads((base/'block_s4_n23.json').read_text())['rows'][255]
independent=json.loads(Path(__file__).with_name('training_neighbor_review_results.json').read_text())
assert independent['model_sha256']==raw[-1]
result=dict(seeds=out,unique_models=len(models),panel_hashes=hashes,
            heldout_native_accuracy=independent['accuracy'],heldout_polygon_accuracy=raw[1],
            heldout_absolute_difference=float(abs(D(independent['accuracy'])-D(str(raw[1])))),
            cpu_seconds=time.process_time()-start,
            scope='Independent raw point ratios, model hashes and preselected separate geometry; no CI coverage certification.')
Path(__file__).with_name('training_summary_review_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='panel_hashes'},indent=2))
