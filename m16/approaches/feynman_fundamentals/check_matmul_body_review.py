import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
import re,json,hashlib,time
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/log_cofactor_cost/matmul');start=time.process_time()
def instructions(name,lo,hi):
 rows=[]
 for line in (P/name).read_text().splitlines():
  m=re.match(r'\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+)+\s*(\w+)\s*(.*)',line)
  if m and lo<=int(m[1],16)<=hi:rows.append((int(m[1],16),m[2],m[3]))
 return rows
nn='small_b0_nn_disassembly.txt';packed='dgemm_kernel_SKYLAKEX_disassembly.txt'
for lo,hi,count in [(0xa89589,0xa895ab,6),(0xa897a9,0xa897af,2)]:
 rows=instructions(nn,lo,hi);v=[r for r in rows if r[1]=='vfmadd231pd'];assert len(v)==count and all('%zmm' in r[2] and '%k' not in r[2] for r in v)
for lo,hi,count in [(0xa895bb,0xa895e6,6),(0xa897bf,0xa897cf,2)]:
 v=[r for r in instructions(nn,lo,hi) if r[1]=='vmulpd'];assert len(v)==count and all('%zmm' in r[2] and '%k' not in r[2] for r in v)
for lo,hi,op in [(0xa452be,0xa452d3,'vfmadd231pd'),(0xa452ec,0xa45319,'vfmadd213pd')]:
 v=[r for r in instructions(packed,lo,hi) if r[1]==op];assert len(v)==4 and all('%ymm' in r[2] and '%k' not in r[2] for r in v)
assert 166*6+2*2==1000
out=dict(small_nn_per_column=4*8*2+8,small_nn_N4=4*(4*8*2+8),small_nn_N1000=1000*(4*8*2+8),packed_4x4_body=4*4*4*2+4*4*2,symmetric_triangular_additions=10,symmetric_guard_envelope=11,symmetric_category_envelope=181,pins={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in [nn,packed,'syrk_kernel_L_disassembly.txt','syrk_LT_disassembly.txt','dscal_k_SKYLAKEX_disassembly.txt','NN_PRICE.md','SYRK_PRICE.md']},review_cpu=time.process_time()-start,scope='Saved reached unmasked SIMD arithmetic bodies independently inspected/contracted, conditional on reviewed runtime target/shapes and branch/layout applicability. NN applies onlyM4K4 evenN4/1000; SYRK packedbody160+triangle10+guardupper11. No price for NT/ordinary fallback, no reduction/libm/control/memory/full-native claim; source array layout application remains separate.')
Path(__file__).with_name('matmul_body_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='pins'}))
