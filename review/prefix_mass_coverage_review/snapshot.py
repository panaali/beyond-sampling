"""Byte-only Git snapshot for T231; never decode NPZ or import author code."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parent
COMMITS={'width':'4b6bbd651ac4fff69e2884b10ff474a7425f796f','gpu':'f02482d6e30502d94a2abd27ce09dc3167cbb6a3'}
W='m16/approaches/width16_probe/'
G='m16/approaches/gpu_cost_law/'
SOURCES={
 'width':[W+n for n in ['README.md','coverage.py','coverage.json','eventword.py','prefix.py','estT.py','zoo44.py','drive6.sh','drive6.log']]+['m16/SCALING_OBSTRUCTIONS.md','m16/approaches/arc_m43_repro/m43.py','m16/approaches/arc_m43_repro/m43_weights.npz']+[W+f'eventword/{m}_k{k}.{ext}' for m in ['target','zoo16_4','zoo16_5'] for k in [2,3,4] for ext in ['json','npz']],
 'gpu':[G+n for n in ['README.md','prefix.py','gmass.py','estG.py','estT_base.py','geomfast.py','orth6.py','colab/q_pmass.sh','colab/runq.sh']]+['m16/approaches/arc_m43_repro/m43.py','m16/approaches/arc_m43_repro/m43_weights.npz']+[G+f'results/prefix/k{k}.json' for k in [2,3,4,5]]+[G+f'results/prefix/k{k}{tail}' for k in [3,4,5] for tail in ['_masses.json','_regionmass.npz']]
}
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def main():
 assert not (ROOT/'manifest.json').exists() and not (ROOT/'frozen').exists()
 rows=[]
 for tag,paths in SOURCES.items():
  assert len(paths)==len(set(paths))
  for path in paths:
   b=git('show',COMMITS[tag]+':'+path);blob=git('rev-parse',COMMITS[tag]+':'+path).decode().strip();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==blob
   p=ROOT/'frozen'/tag/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
   rows.append(dict(tag=tag,commit=COMMITS[tag],path=path,snapshot=str(p.relative_to(ROOT)),git_blob=blob,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
 oldpath=Path('/workspace/work/recover-small_h2_34/review/width16_structure_review/manifest.json');old=json.loads(oldpath.read_text());prior={x['path']:x for x in old['files']};same=[]
 for r in rows:
  if r['tag']=='width' and '/eventword/' in r['path'] and r['path'].endswith('.npz'):
   assert r['sha256']==prior[r['path']]['sha256'] and r['git_blob']==prior[r['path']]['git_blob'];same.append(r['path'])
 assert len(same)==9
 extra=[oldpath,oldpath.with_name('results.json'),Path('/workspace/work/feynman_fundamentals/m16/approaches/feynman_fundamentals/t228_saved_review.json')]
 manifest=dict(task='T231 independent T222 evidence',claim='7316c33f',commits=COMMITS,records=rows,prior_T228_identical_nine_arrays=same,prior_pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in extra},scope='Source/text and byte hashing/copy only; no archive decoding, author imports, mass/geometry/model/RNG calls. Current Git identity is not historical runtime/checkpoint identity.')
 (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(dict(payloads=len(rows),bytes=sum(r['bytes'] for r in rows),NPZ_arrays_decoded=0)))
if __name__=='__main__':main()
