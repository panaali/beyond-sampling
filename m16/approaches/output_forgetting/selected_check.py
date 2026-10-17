from pathlib import Path
import json,time
from selected import selected

def main():
    start=time.process_time();p=Path(__file__).parent
    controls=json.loads((p/'signed_controls_results.json').read_text())['rows']
    trained=json.loads((p/'signed_pilot_results.json').read_text())['quadrature']
    rows=[]
    for source,name in ((controls,'controls'),(trained,'trained')):
        for old in source:
            if old['outer']!=128 or old['inner']!=64:continue
            for outer in (64,128):
                r=selected(old['T'],old['k'],old['signs'],outer);r['source']=name
                r['reference']=old['classes'][3 if old['signs'][1]=='+' else 1]
                r['difference']=r['correct_mass']-r['reference'];rows.append(r)
    # Fresh construction on every call; no rule or model-specific advice cache.
    timings=[]
    for _ in range(20):
        stamp=time.process_time();mass=0.;ss=[]
        for old in trained:
            if old['outer']!=128 or old['inner']!=64:continue
            r=selected(10,old['k'],old['signs'],64);mass+=r['correct_mass'];ss.append(r['stats'])
        timings.append(time.process_time()-stamp)
    out={'rows':rows,'fresh_union_cpu_seconds':timings,'fresh_union_mass':mass,'fresh_union_stats':ss,'cpu_seconds':time.process_time()-start,'no_model_specific_cache':True,'portable_cost_audited':False}
    (p/'selected_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'cpu_seconds':out['cpu_seconds'],'max_abs_difference':max(abs(r['difference']) for r in rows),'trained_max_abs_difference':max(abs(r['difference']) for r in rows if r['source']=='trained'),'fresh_union_mass':mass,'fresh_union_mean_cpu':sum(timings)/len(timings),'fresh_union_stats':ss},indent=2))
if __name__=='__main__':main()
