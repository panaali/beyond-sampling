"""Independent result arithmetic and a controlled suffix precision intervention."""
import decimal, hashlib, importlib.util, json, sys, time
from pathlib import Path
decimal.getcontext().prec=60
D=decimal.Decimal
here=Path(__file__).resolve().parent
measure=Path('/workspace/work/recover-small_h4_4/m16/approaches/rank_conditioned_transport/measure_cycle/pilot.json')
data=json.loads(measure.read_text(),parse_float=D)
rows=[]
for row in data['rows']:
    total=sum(row['stats']['contributions'].values(),D(0))
    rows.append({'coordinate':row['coordinate'],'order':row['order'],'sum':str(total),
                 'reported_sum_difference':str(total-row['estimate']),
                 'reference_error':str(total-data['reference'])})
values={(r['coordinate'],r['order']):D(r['sum']) for r in rows}
arithmetic={'rows':rows,'q_order_change':str(values['q',4]-values['q',2]),
            'u_order_change':str(values['u',4]-values['u',2]),
            'gl4_coordinate_difference':str(values['u',4]-values['q',4])}

p=Path('/workspace/work/recover-small_h2_910/m16/approaches/three_suffix_reset')
sys.path.insert(0,str(p))
spec=importlib.util.spec_from_file_location('review_suffix_precision',p/'integrate.py')
engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
original_rule,original_older=engine.rule,engine.older
def rule(n,a=0.,b=1.):
    return original_rule(64 if n==32 and a==0. and b==8. else n,a,b)
def older(suffix,n=32):
    return original_older(suffix,64)
engine.rule,engine.older=rule,older
tick=time.process_time()
result=engine.integrate(engine.cells(),16)
result['review_cpu_seconds']=time.process_time()-tick
baseline=next(r for r in json.loads((p/'integration.json').read_text())['runs'] if r['order']==16)
result['baseline_contribution']=baseline['contribution']
result['difference']=result['contribution']-baseline['contribution']
result['per_label_differences']={str(k):v-baseline['bylabel'][str(k)] for k,v in result['bylabel'].items()}
result['source_sha256']=hashlib.sha256((p/'integrate.py').read_bytes()).hexdigest()
result['scope']='same geometry/engine, independent precision intervention only; not independent topology implementation'
out={'measure_arithmetic':arithmetic,'radial64_target64_angular16':result}
(here/'recovered_precision_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
