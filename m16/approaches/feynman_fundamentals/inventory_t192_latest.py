"""Read-only Git/file-presence inventory; no author execution or numerical review."""
from pathlib import Path
import subprocess,hashlib,json,time
H=Path(__file__).resolve().parent
REV='a5ab8757';BASE='m16/approaches/gpu_cost_law'
def git(*args):return subprocess.check_output(['git',*args])
def main():
 start=time.process_time();commit=git('rev-parse',REV).decode().strip();paths=git('ls-tree','-r','--name-only',commit,BASE).decode().splitlines()
 selected=[f'{BASE}/{p}' for p in ['export_simplices.py','prefix.py','README.md','results/prefix/k2.json','results/prefix/k3.json','results/prefix/k4.json','results/prefix/k5.json']]
 pins={}
 for p in selected:
  b=git('show',commit+':'+p);pins[p]=dict(git_blob=git('rev-parse',commit+':'+p).decode().strip(),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
 roots=[Path('/workspace'),Path('/workspace/clean-neurons')]+sorted(p for p in Path('/workspace/work').iterdir() if p.is_dir())
 checked=[];found=[]
 for root in roots:
  p=root/BASE/'results/simplices';checked.append(str(p))
  if p.is_dir():found.extend(str(q) for q in p.iterdir() if q.is_file())
 out=dict(complete=True,revision=commit,tracked_folder_paths=len(paths),tracked_numpy_archives=[p for p in paths if p.endswith(('.npy','.npz'))],expected_local_directories_checked=checked,files_found_at_expected_export_paths=found,pins=pins,review_cpu=time.process_time()-start,new_author_geometry_mass_MC_calls=0,
 conclusions=['Exporter source exists, but this revision has no tracked NumPy arrays within gpu_cost_law.','No simplex payload was found at the expected export location in the enumerated workspace roots. This is not a statement about remote Colab, arbitrary external paths, ignored archives elsewhere, or future revisions.','T193 fixed-cell numerical work remains blocked until actual arrays and accepted provenance arrive. Source code alone does not clear that dependency.'],
 coordination={'recover_34':'Distinct exporter source/schema/provenance prerequisite, no capture or mass execution.','recover_78':'Distinct S4 prefix count/source/extrapolation review after prior handoff; existing T55 T192 audit does not cover new counts.'},
 scientific_limits=['No acceptance of S4 counts, extrapolation, exporter mass references, or new MC truth.','Tracked k2..k5 summaries and prefix source are inventoried, not independently verified here.','Existing T192 cost and parity qualifications remain.'])
 (H/'t192_latest_inventory.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['revision','tracked_folder_paths','tracked_numpy_archives','files_found_at_expected_export_paths','new_author_geometry_mass_MC_calls']}))
if __name__=='__main__':main()
