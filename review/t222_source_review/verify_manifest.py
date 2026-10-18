"""T231 independent byte/Git provenance check; no archive decoding or author imports."""
import hashlib,json,pathlib,subprocess,time,traceback
HERE=pathlib.Path(__file__).resolve().parent
AUTHOR=pathlib.Path('/workspace/work/recover-small_h4_4/review/prefix_mass_coverage_review')
OUT=HERE/'manifest_review.json'
state={'status':'started','records':[],'scope':'Bytes/Git metadata only; no NPZ decoding, array arithmetic, author imports, model/geometry/mass/RNG calls.'}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save():
 t=OUT.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(OUT)
assert not OUT.exists();start=time.process_time();save()
try:
 m=json.loads((AUTHOR/'manifest.json').read_text());assert len(m['records'])==51
 seen=set()
 for r in m['records']:
  key=(r['tag'],r['path']);assert key not in seen;seen.add(key)
  assert r['commit']==m['commits'][r['tag']]
  p=AUTHOR/r['snapshot'];assert p.is_relative_to(AUTHOR/'frozen'/r['tag'])
  b=p.read_bytes();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
  blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();assert blob==r['git_blob']
  g=subprocess.check_output(['git','show',f"{r['commit']}:{r['path']}"]);assert g==b
  state['records'].append({'tag':r['tag'],'path':r['path'],'sha256':r['sha256'],'git_blob':blob,'bytes':len(b),'git_bytes_equal':True});save()
 assert sum(r['bytes'] for r in state['records'])==7926138
 for path,h in m['prior_pins'].items():assert digest(pathlib.Path(path))==h
 oldpath=pathlib.Path('/workspace/work/recover-small_h2_34/review/width16_structure_review/manifest.json')
 old=json.loads(oldpath.read_text());lookup={r['path']:r for r in old['files']}
 assert len(m['prior_T228_identical_nine_arrays'])==9 and len(set(m['prior_T228_identical_nine_arrays']))==9
 match=[]
 for path in m['prior_T228_identical_nine_arrays']:
  a=next(r for r in m['records'] if r['tag']=='width' and r['path']==path);b=lookup[path]
  assert all(a[k]==b[k] for k in ['bytes','sha256','git_blob'])
  match.append(path)
 review=pathlib.Path('/workspace/work/feynman_fundamentals/m16/approaches/feynman_fundamentals/t228_saved_review.json')
 prior=json.loads(review.read_text());assert prior['complete'] is True and prior['pass_all'] is True and prior['arrays']==9
 state.update(status='passed',count=51,total_bytes=7926138,prior_nine_matches=match,prior_pins=m['prior_pins'],pins={str(AUTHOR/f):digest(AUTHOR/f) for f in ['manifest.json','THEORY.md','SOURCE_REVIEW.md','README.md']})
except BaseException as e:state.update(status='failed',error=repr(e),traceback=traceback.format_exc());raise
finally:state['cpu_seconds_excluding_git_children']=time.process_time()-start;save()
