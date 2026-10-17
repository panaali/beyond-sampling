"""Report complete/partial fixed panels without hiding failed or missing rows."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    out=dict(by_H={},complete=True,total_cpu_seconds=0.)
    ref=json.loads((HERE/'suite_reference.json').read_text())
    for h in [4,5,6]:
        chunks=[json.loads(p.read_text()) for p in sorted(HERE.glob(f'suite_h{h}_*.json'))]
        rows=[r for c in chunks for r in c['models']];names=[r['name'] for r in rows]
        assert len(names)==len(set(names)),'Overlapping raw chunks'
        complete=len(rows)==157 and set(names)==set(ref[str(h)]['rows']) and all(c['complete'] for c in chunks)
        cpu=sum(c['cpu_seconds_including_imports'] for c in chunks);out['total_cpu_seconds']+=cpu;out['complete']&=complete
        stats=dict(complete=complete,models=len(rows),chunks=len(chunks),cpu_seconds=cpu,methods={},failures=[])
        stats['groups']={}
        for group in sorted({'seeds' if r['name'].startswith('seed') else r['name'].rsplit('_',1)[0] for r in rows}):
            selected=[r for r in rows if ('seeds' if r['name'].startswith('seed') else r['name'].rsplit('_',1)[0])==group]
            stats['groups'][group]={}
            for m in ['reference','omit','capped']:
                valid=[r for r in selected if 'estimate'in r['methods'].get(m,{})]
                if not valid:continue
                mse=sum(r['methods'][m]['bias']**2 for r in valid)/len(valid)
                sample=sum(r['truth']*(1-r['truth'])*3*h*(h+2)/r['methods'][m]['proxy_ops'] for r in valid)/len(valid)
                stats['groups'][group][m]=dict(models=len(valid),mean_squared_error=mse,mean_sampling_proxy_variance=sample,ratio=mse/sample)
        for m in ['reference','omit','capped']:
            valid=[]
            for r in rows:
                v=r['methods'].get(m,{})
                if 'estimate'not in v:stats['failures'].append(dict(model=r['name'],method=m,detail=v));continue
                valid.append(v)
            if valid:
                stats['methods'][m]=dict(models=len(valid),mean_ops=sum(v['proxy_ops'] for v in valid)/len(valid),
                    max_absolute_bias=max(abs(v['bias']) for v in valid),max_ratio=max(v['mse_ratio_proxy'] for v in valid),
                    fail_count=sum(v['mse_ratio_proxy']>=1 for v in valid))
        pairs=[r for r in rows if all('estimate'in r['methods'].get(m,{}) for m in ['reference','omit'])]
        if pairs:
            savings=[1-r['methods']['omit']['proxy_ops']/r['methods']['reference']['proxy_ops'] for r in pairs]
            changes=[dict(name=r['name'],delta=r['methods']['omit']['estimate']-r['methods']['reference']['estimate']) for r in pairs if abs(r['methods']['omit']['estimate']-r['methods']['reference']['estimate'])>1e-12]
            stats.update(savings=dict(min=min(savings),mean=sum(savings)/len(savings),max=max(savings)),changed=changes,
                max_truth_difference=max(abs(r['truth_difference']) for r in rows),
                max_prior_reference_difference=max(abs(r['reference_difference_from_prior']) for r in rows if 'reference_difference_from_prior'in r))
        out['by_H'][h]=stats
    (HERE/'suite_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({**out,'by_H':{h:{k:v for k,v in s.items() if k not in ['changed','groups']} for h,s in out['by_H'].items()}},indent=2))


if __name__=='__main__':main()
