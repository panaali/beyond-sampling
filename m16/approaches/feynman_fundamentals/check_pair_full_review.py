"""Independent Decimal contraction and metrics for the four T118 raw rows."""
from decimal import Decimal as D,getcontext
from pathlib import Path
import json,time
getcontext().prec=60
start=time.process_time()
p=Path('/workspace/work/recover-small_h4_4/m16/approaches/gaussian_pair_terminal/full_cycle/pilot.json')
d=json.loads(p.read_text(),parse_float=D);rows={};out=[]
for r in d['rows']:
    s=sum(r['stats']['contributions'].values(),D(0));e=s-d['reference']
    assert abs(s-r['estimate'])<D('1e-15')
    l2=sum(x*x for x in r['per_label_errors']).sqrt()
    row={'paired':r['paired'],'order':r['order'],'estimate':str(s),'error':str(e),
         'component_l2':str(l2),'error_squared_cpu':str(e*e*r['stats']['cpu_seconds'])}
    rows[r['paired'],r['order']]=(r,s,e,l2);out.append(row)
b=rows[False,4];n=rows[True,4]
assert b[0]['stats']['contributions']['2']==n[0]['stats']['contributions']['2']
result={'rows':out,'baseline_gl2_to_gl4':str(b[1]-rows[False,2][1]),'paired_gl2_to_gl4':str(n[1]-rows[True,2][1]),
        'absolute_error_improvement':str(abs(b[2]/n[2])),'component_l2_improvement':str(b[3]/n[3]),
        'error_squared_cpu_improvement':str(b[2]**2*b[0]['stats']['cpu_seconds']/(n[2]**2*n[0]['stats']['cpu_seconds'])),
        'unchanged_penultimate_label':True,'cpu_seconds':time.process_time()-start,
        'scope':'one fixed h3 model; both frozen refinement gates fail; no sampling or certified cost comparison'}
Path(__file__).with_name('pair_full_review_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
