import json,resource,time,traceback
from common import *

def check(model,power,seed,copy):
    row=dict(power=power,training_seed=seed,copy=copy,model_sha256=model_hash(model));row['candidate']=estimate(model);row['truth']=truth(model)
    if copy in (-1,0):
        row['original_truth']=truth(model,strict=False);row['independent']=independent(model);row['fully_resolved']=estimate(model,K=(4,4),rule='impact',eps=.1)
        for key in ('original_truth','independent','fully_resolved'):assert abs(row[key]['value']-row['truth']['value'])<1e-12,(power,seed,copy,key)
        assert abs(row['independent']['partition']-1)<1e-12
    p=row['truth']['value'];row['error']=row['candidate']['value']-p;row['ratio']=row['error']**2/(p*(1-p)*72/row['candidate']['operations'])
    return row

def run():
    start=time.process_time();out=manifest();out['rows']=[];past=json.loads((SOURCE/'m16/approaches/understand_m43/eval75.json').read_text())
    for power in (30,29):
        for seed in range(5):
            model=load(power,seed);row=check(model,power,seed,-1)
            if power==30:
                expected=past['zoo'][str(seed)]['clean'];assert abs(row['candidate']['value']-expected['t75'])<1e-12;assert row['candidate']['operations']==expected['t75_ops'];assert abs(row['truth']['value']-expected['truth'])<1e-12
            out['rows'].append(row)
            if power==29:
                for i,noisy in enumerate(noisy_copies(model,4,.05,20)):out['rows'].append(check(noisy,power,seed,i))
            out.update(cpu_seconds=time.process_time()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            (HERE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n');print(power,seed,'clean truth',row['truth']['value'],'error',row['error'],'ratio',row['ratio'],flush=True)
    (HERE/'manifest.json').write_text(json.dumps(manifest(),indent=2)+'\n')
if __name__=='__main__':
    try:run()
    except Exception as exc:(HERE/'pilot_failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2)+'\n');raise
