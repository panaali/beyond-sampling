"""Saved-data pilot summary only; no estimator or reference replay."""
from pathlib import Path
from collections import Counter
import json,math
P=Path(__file__).resolve().parent

def main():
 d=json.loads((P/'pilot.json').read_text());p=json.loads((P/'protocol.json').read_text());r=d['rows']
 assert d['complete'] and [x['index'] for x in r]==p['indices'] and len(r)==64
 s=dict(covered=len(r),error_sum=math.fsum(x['error'] for x in r),absolute_error_sum=math.fsum(abs(x['error']) for x in r),max_abs_error=max(abs(x['error']) for x in r),bound_sum=math.fsum(x['bound'] for x in r),reference_bound_sum=math.fsum(x['reference_bound'] for x in r),bounds_over_target=sum(x['bound']>d['per_cone_target'] for x in r),comparison_failures=sum(not x['comparison_pass'] for x in r),modes=d['methods'],center_orders=dict(Counter(x['center'].get('order') for x in r if x['center']['mode']=='quadrature')),rescue_rows=[{k:x[k] for k in ('index','value','bound','error','cpu')}|dict(order=x['rescue']['order'],stages=[a['method'] for a in x['rescue']['stages']],exact_inventory=x['rescue']['exact_operation_inventory']) for x in r if 'rescue' in x],float_named_total=d['known_float_named_total'],cpu=d['cpu_including_imports'],scope='Fixed64 only; no full accuracy or cost extrapolation, full FP or portable-cost certificate.')
 (P/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps({k:v for k,v in s.items() if k!='rescue_rows'}))
if __name__=='__main__':main()
