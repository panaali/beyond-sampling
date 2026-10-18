"""Inventory saved printed evidence only; no estimates regenerated."""
import hashlib,json,re,math
from collections import defaultdict
from pathlib import Path
HERE=Path(__file__).resolve().parent
SRC=Path('/workspace/work/admin-recover56-cumulant-review/review/arc_cumulants_negative')
manifest=json.loads((HERE/'source_manifest.json').read_text())
for r in manifest['files']:
    p=SRC.parents[1]/r['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
pattern=re.compile(r'^T=(\d+)\s+(\S+)\s+truth ([\d.]+)\s+(\S+)\s+([-+\d.]+)\s+\(([-+\d.]+) pp\)\s+cost ~([\d,]+) FPE$')
outputs=[];rows=[]
for p in sorted(SRC.glob('out_*.txt')):
    lines=p.read_text().splitlines();data=[x for x in lines if x.strip()]
    matches=[]
    for i,line in enumerate(lines,1):
        m=pattern.match(line.strip())
        if m:
            t,name,truth,variant,est,pp,cost=m.groups();r=dict(T=int(t),model=name,variant=variant,truth_printed=float(truth),estimate_printed=float(est),bias_pp_printed=float(pp),rough_fpe_printed=int(cost.replace(',','')),path=p.name,line=i);rows.append(r);matches.append(r)
    outputs.append(dict(path=p.name,lines=len(lines),last_lines=data[-3:],variant_rows=len(matches),contains_nonfinite_text=bool(re.search(r'\b(?:nan|inf)\b',p.read_text(),flags=re.I))))
unique={};duplicates=[]
for r in rows:
    k=(r['T'],r['model'],r['variant'])
    if k in unique:
        assert all(r[f]==unique[k][f] for f in ('truth_printed','estimate_printed','bias_pp_printed','rough_fpe_printed'))
        duplicates.append(dict(key=k,first=[unique[k]['path'],unique[k]['line']],duplicate=[r['path'],r['line']]))
    else:unique[k]=r
by=defaultdict(list)
for r in unique.values():by[(r['T'],r['variant'])].append(r)
groups=[]
for (t,v),rs in sorted(by.items()):groups.append(dict(T=t,variant=v,models=len(rs),model_names=sorted(r['model'] for r in rs),rmse_from_rounded_printed_estimates=math.sqrt(sum((r['estimate_printed']-r['truth_printed'])**2 for r in rs)/len(rs)),qualification='Subset descriptive RMSE from rounded printed values; no raw-copy bootstrap or common-cost confirmation.'))
placeholders=['CEIL10_G','CEIL10_E3','CEIL10_E4','R4S_2','R4S_3','R4S_4','R10U_2','R10U_3','J10G','J10GC','C10G','C10GC','C10A','C10AC','ARCCELL','ARCCELLC','RMSE_TABLE']
readme=(SRC/'README.md').read_text()
out=dict(complete_inventory=True,output_files=outputs,variant_rows=rows,unique_variant_rows=list(unique.values()),duplicates=duplicates,groups=groups,remaining_readme_placeholders=[s for s in placeholders if s in readme],scope='Evidence inventory only; headers without result lines are not completed experiments. Duplicate printed rows are not independent replicates.',old_numerical_runs_repeated=False)
(HERE/'inventory.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(output_files=len(outputs),printed_rows=len(rows),unique_rows=len(unique),duplicate_rows=len(duplicates),groups=groups)))
