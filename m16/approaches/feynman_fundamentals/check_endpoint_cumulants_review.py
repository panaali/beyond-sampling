"""Independent cumulants via set partitions, not central-moment expansion."""
from decimal import Decimal as D, getcontext
from itertools import product
from math import factorial
from pathlib import Path
import hashlib,json,time
start=time.process_time();getcontext().prec=60
path=Path('/workspace/work/recover-small_h2_910/m16/approaches/endpoint_cumulant_review/trained_t3.json')
data=json.loads(path.read_text(),parse_float=D)
def partitions(items):
    if not items:
        yield []
        return
    first,*rest=items
    for part in partitions(rest):
        yield [(first,)]+part
        for i in range(len(part)):
            yield part[:i]+[(first,)+part[i]]+part[i+1:]
def at(array,ids):
    for i in ids:array=array[i]
    return array
rows=[]
for endpoint in data['endpoint_cases']:
    for case in endpoint['conditional']:
        label=case['target'];mass=case['law_mass'];degree_errors={}
        for degree in range(1,5):
            largest_error=D(0);largest_value=D(1)
            for ids in product(range(3),repeat=degree):
                value=D(0)
                for partition in partitions(list(range(degree))):
                    term=D((-1)**(len(partition)-1)*factorial(len(partition)-1))
                    for block in partition:
                        term*=at(data['joint_moments'][str(len(block))][label],tuple(ids[i] for i in block))/mass
                    value+=term
                author=at(case['cumulants'][str(degree)],ids)
                largest_error=max(largest_error,abs(value-author));largest_value=max(largest_value,abs(value))
            degree_errors[str(degree)]=dict(maximum_absolute_difference=float(largest_error),
                                           scaled_difference=float(largest_error/largest_value))
        rows.append(dict(atom_separated=endpoint['atom_separated'],target=label,degree_errors=degree_errors))
maximum=max(v['scaled_difference'] for row in rows for v in row['degree_errors'].values())
assert maximum<1e-12
out=dict(rows=rows,maximum_scaled_difference=maximum,cpu_seconds=time.process_time()-start,
         artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
         scope='Independent cumulant algebra on saved numerical raw moments, not an independent moment enclosure or endpoint implementation.')
Path(__file__).with_name('endpoint_cumulants_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
