"""Pinned assembly/source-path contraction; no kernel or model execution."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import time

HERE=Path(__file__).resolve().parent
OUT=HERE/'instruction_ledger.json'
assert not OUT.exists(),'Preserve evidence'
t0=time.process_time()
build=json.loads((HERE/'build1/build_manifest.json').read_text())
b=(HERE/'build1/disassembly.txt').read_bytes()
assert hashlib.sha256(b).hexdigest()==build['disassembly_sha256']
text=b.decode().split('<guard3>:',1)[1].split('<guard_getcsr>:',1)[0]
instructions={}
for line in text.splitlines():
    m=re.match(r'\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+)+\s*([a-z][a-z0-9]*)\s*(.*)',line)
    if m: instructions[int(m[1],16)]=(m[2],m[3])
arith={'mulsd','addsd','subsd'}
body=Counter(op for adr,(op,args) in instructions.items() if 0x1232<=adr<=0x130b and op in arith)
outward=Counter(op for adr,(op,args) in instructions.items() if 0x1370<=adr<0x13db and op in arith)
assert body=={'mulsd':19,'subsd':4,'addsd':6}
assert outward=={'subsd':1,'addsd':1}
assert not any('fmadd' in op or op in {'mulpd','addpd','subpd'} for op,args in instructions.values())
for address,opcode in [(0x11a6,'andpd'),(0x11aa,'ucomisd'),(0x1203,'ucomisd'),
                        (0x120b,'andpd'),(0x1218,'comisd'),(0x11ea,'comisd'),
                        (0x130f,'ucomisd'),(0x1317,'ucomisd')]:
    assert instructions[address][0]==opcode
calls=[args for op,args in instructions.values() if op=='call']
assert sum('nextafter@plt' in s for s in calls)==2
assert sum('fegetround@plt' in s for s in calls)==1

control=json.loads((HERE/'results.json').read_text())
rows=[]
for saved in control['rows']:
    a=[float.fromhex(x) for row in saved.get('input_hex',saved.get('minor_hex')) for x in row]
    status=saved['status']
    fp=Counter();fp_compare=9;mask128=9;nextafter=0
    range_compare=0
    if status!=2:
        assert all(math.isfinite(x) for x in a)
        mask128+=9
        # Saved assembly loops all9 inputs: equality, then low and conditional high.
        for x in a:
            range_compare+=1
            if x!=0:
                range_compare+=1
                if abs(x)>=2.**-100: range_compare+=1
        fp_compare+=range_compare
    if status==0:
        fp.update(body)
        fp_compare+=1
        error=float.fromhex('0x0p0') if saved['bits'][4]=='0000000000000000' else 1
        if error==0:
            fp_compare+=1
        else:
            fp.update(outward);nextafter=2
    rows.append({'name':saved['name'],'status':status,'scalar_fp_arithmetic':dict(fp),
                 'scalar_fp_arithmetic_total':sum(fp.values()),'scalar_fp_comparisons':fp_compare,
                 'range_fp_comparisons':range_compare,'packed128_bitmask_instructions':mask128,
                 'external_nextafter_calls':nextafter,'external_fegetround_calls':1})
totals={key:sum(row[key] for row in rows) for key in [
    'scalar_fp_arithmetic_total','scalar_fp_comparisons','packed128_bitmask_instructions',
    'external_nextafter_calls','external_fegetround_calls']}
out={'rows':rows,'totals':totals,'common_scalar_arithmetic_body':dict(body),'outward_scalar_arithmetic':dict(outward),
     'disassembly_sha256':build['disassembly_sha256'],
     'control_results_sha256':hashlib.sha256((HERE/'results.json').read_bytes()).hexdigest(),
     'cpu_seconds':time.process_time()-t0,
     'scope':'Source-path counts deduced from pinned assembly and fixed saved inputs, not instruction tracing or timing. Packed ANDPD is a128bit mask instruction, not two floating operations. External function bodies, integer/control/memory and Python exact fallback excluded. Four rejected environment probes excluded from89-row sum. No totalcost/speed or104-envelope comparison.'}
with OUT.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({'totals':totals,'cpu_seconds':out['cpu_seconds']},indent=2))
