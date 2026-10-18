"""Independent source snapshot identity check; no author imports or model/geometry calls."""
from pathlib import Path
import subprocess,json,hashlib,time
P=Path('/workspace/work/recover-small_h2_34/review/t193_export_contract');H=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
 start=time.process_time();m=json.loads((P/'manifest.json').read_text());repo=P.parents[1];r=m['source_commit'];checked=[]
 for f in m['files']:
  b=subprocess.check_output(['git','show',r+':'+f['path']],cwd=repo);snap=(repo/f['snapshot']).read_bytes()
  assert b==snap and len(b)==f['bytes'] and sha(b)==f['sha256']
  assert subprocess.check_output(['git','rev-parse',r+':'+f['path']],cwd=repo).decode().strip()==f['git_blob'];checked.append(f['path'])
 f=m['incidental_binary'];b=subprocess.check_output(['git','show',r+':'+f['path']],cwd=repo);assert len(b)==f['bytes'] and sha(b)==f['sha256']
 assert subprocess.check_output(['git','rev-parse',r+':'+f['path']],cwd=repo).decode().strip()==f['git_blob']
 schema=json.loads((P/'schema.json').read_text());assert schema['missing_arrays_block_numerical_work'] and not schema['executed_validator'] and schema['expected_seed_ids']==[0,2]
 out=dict(complete=True,pass_all=True,source_commit='25633ca0',author_branch_commit=r,checked_text_snapshots=checked,ancillary_binary_hashed_not_decoded=f['path'],pins={n:sha((P/n).read_bytes()) for n in ['SOURCE.md','ACCEPTANCE.md','schema.json','manifest.json']},checker_sha256=sha(Path(__file__).read_bytes()),review_cpu=time.process_time()-start,new_author_import_model_geometry_mass_calls=0,
 findings=['Error-simplex selection is global emission indices0,stride,...; K=ceil(total/stride) only for positive stride and completed source execution. Per-half ownership is not saved.','Rays are row generators; exact stored binary entries must be preserved. Floating unit normalization and absdet are not exact certificates.','Model naming alone does not bind checkpoint bytes; fast geometry/rules/device/environment/source versions affect provenance.','Conditional SLIVER hazard verified by source: masses_each forwards batched _orth5; non-None SLIVER zeros thin entries and puts summed _thin_split result in the first thin row. Per-parent semantics are not preserved in general. Actual occurrence remains unknown without run environment/payload.','Reference singleton loop avoids cross-parent aggregation but shares numerical family; maxlev24 does not certify threshold attainment.','Source writes archive at end and does not itself establish local resource/durable-run protocol.'],
 disposition='Source-derived acceptance contract passes. No exporter execution, regenerated data, or T193 numerical work authorized. Actual arrays remain missing; incomplete metadata must remain explicit. Original author branch ancestry remains separate from owned snapshot parity.')
 (H/'t215_source_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(complete=True,text_files=len(checked),review_cpu=out['review_cpu'])))
if __name__=='__main__':main()
