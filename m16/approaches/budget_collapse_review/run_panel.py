"""Frozen per-T production cycle; persist each rule and enforce bounded CPU."""
import time
START=time.process_time()
import argparse,hashlib,json,resource
from pathlib import Path
import engine as E

HERE=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--T',type=int,choices=[4,5,6],required=True)
    args=ap.parse_args();path=HERE/f'panel_T{args.T}.json';planpath=HERE/'calibration.json'
    plans=json.loads(planpath.read_text());engine_hash=hashlib.sha256((HERE/'engine.py').read_bytes()).hexdigest()
    assert engine_hash==plans['engine_sha256']
    phash=hashlib.sha256(planpath.read_bytes()).hexdigest()
    out=json.loads(path.read_text()) if path.exists() else dict(T=args.T,models=[],cycles=[],complete=False,
        plan_sha256=phash,engine_sha256=engine_hash,source_pins=E.PINS,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    assert out['plan_sha256']==phash and out['engine_sha256']==engine_hash
    table_start=time.process_time();E.C.tables(args.T);table_cpu=time.process_time()-table_start
    newrules=0;stop=False
    for plan in [p for p in plans['models'] if p['T']==args.T]:
        sd=plan['seed'];M=E.model(args.T,sd)
        matches=[q for q in out['models'] if q['seed']==sd]
        row=matches[0] if matches else dict(H=16,T=args.T,seed=sd,budgets=[])
        if not matches:out['models'].append(row)
        for target in plan['plans']:
            B=target['budget'];matches=[q for q in row['budgets'] if q['target_budget']==B]
            cell=matches[0] if matches else dict(target_budget=B,feasible=target['feasible'],
                faces_per_kind=target['faces_per_kind'],rules=[])
            if not matches:row['budgets'].append(cell)
            if not target['feasible']:continue
            for r in range(len(cell['rules']),16):
                if time.process_time()-START>90:stop=True;break
                rule=E.faces(M,target['faces_per_kind'],44100+r)
                rule.update(rule_seed=44100+r,pilot_fpe=plan['pilot']['variable_fpe'],
                    setup_fpe=plan['fixed']['setup_fpe'])
                rule['cold_fpe']=rule['variable_fpe']+rule['pilot_fpe']+rule['setup_fpe']
                rule['warm_fpe']=rule['cold_fpe']-plan['fixed']['table_flops']/plan['fixed']['unit']
                cell['rules'].append(rule);newrules+=1
                path.write_text(json.dumps(out,indent=2)+'\n')
            print(args.T,sd,B,len(cell['rules']),'rules',flush=True)
            if stop:break
        if stop:break
    out['complete']=len(out['models'])==5 and all(len(m['budgets'])==2 and all(
        not b['feasible'] or len(b['rules'])==16 for b in m['budgets']) for m in out['models'])
    out['cycles'].append(dict(cpu_seconds_including_imports=time.process_time()-START,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,new_rules=newrules,
        table_cpu_seconds=table_cpu,stopped_at_deadline=stop))
    path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(complete=out['complete'],cycle=out['cycles'][-1]),indent=2))


if __name__=='__main__':main()
