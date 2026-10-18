"""Saved S4/noisy-table contraction only; no author imports or numerical enumeration."""
from pathlib import Path
from fractions import Fraction as F
import subprocess,json,hashlib,math,time
H=Path(__file__).resolve().parent;REV='2930bc92';BASE='m16/approaches/gpu_cost_law';pins={}
def blob(path):
 b=subprocess.check_output(['git','show',REV+':'+path]);pins[path]=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b));return b
def read(path):return json.loads(blob(BASE+'/'+path))
def main():
 start=time.process_time();summary=read('results/summary.json');prefix=[read(f'results/prefix/k{k}.json') for k in range(2,6)]
 assert prefix==summary['prefix'] and [r['regions'] for r in prefix]==[34,792,21032,632394]
 assert all(r['complete'] is True and r['subtrees']==r['subtrees_done'] and r['regions']<=r['cells'] for r in prefix)
 growth=[F(b['regions'],a['regions']) for a,b in zip(prefix[:-1],prefix[1:])]
 noisy=summary['noisy'];assert len(noisy)==18;reports=[];raw_count=0
 expected={(h,'0') for h in [8,16]} | {(h,f'0_{kind}{sigma}_{i}') for h in [8,16] for kind,sigma in [('abs','0.05'),('rel','0.005'),('rel','0.01'),('rel','0.02')] for i in [0,1]}
 assert {(r['H'],r['model']) for r in noisy}==expected
 for row in noisy:
  stem=f"results/runs/H{row['H']}T4_{row['model']}_exact_leafsum_truthstart";r=read(stem+'.json');lines=[json.loads(s) for s in blob(BASE+'/'+stem+'.jsonl').decode().splitlines()]
  assert r['H']==row['H'] and r['T']==4 and r['model'].replace(':','_')==row['model'] and r['mode']=='exact' and r['start']=='truth' and r['resumed']==0
  assert r['halves']==r['halves_run']==len(lines)==24 and {v['half'] for v in lines}==set(range(24));lines.sort(key=lambda v:v['half']);raw_count+=len(lines)
  for key in ['cells','cut','resolved','ops','ops_geometry','simplices','thin']:assert sum(v[key] for v in lines)==r[key]
  assert abs(math.fsum(v['sec_geometry'] for v in lines)-r['sec_geometry'])<1e-10
  assert 1-sum(v['err'] for v in lines)==r['acc'] and sum(v['leaf'] for v in lines)-1==r['leafsum_minus_1']
  assert row==dict(H=r['H'],model=r['model'].replace(':','_'),acc=r['acc'],cells=r['cells'],regions=r['regions'],ops=r['ops'])
  clean=next(c for c in noisy if c['H']==row['H'] and c['model']=='0')
  ratios={k:F(int(row[k]),int(clean[k])) for k in ['cells','regions','ops']}
  reports.append(dict(H=row['H'],model=row['model'],ratios={k:str(v) for k,v in ratios.items()},acc=r['acc'],halves=24,regions_from_summary_only=True))
 ranges={k:{'minimum':str(min(F(r['ratios'][k]) for r in reports if r['model']!='0')),'maximum':str(max(F(r['ratios'][k]) for r in reports if r['model']!='0'))} for k in ['cells','regions','ops']}
 for f in ['prefix.py','estP.py','estT_base.py','geomfast.py','summary.py','README.md']:blob(BASE+'/'+f)
 # The revised branch supplies payload bytes; only hash them here, no NumPy decoding.
 exports={}
 for seed in [0,2]:
  path=BASE+f'/results/simplices/H8T5_zoo{seed}.npz';b=blob(path);exports[str(seed)]=pins[path]
 out=dict(complete=True,pass_saved_contractions=True,revision=subprocess.check_output(['git','rev-parse',REV]).decode().strip(),pins=pins,prefix=prefix,prefix_growth_exact=list(map(str,growth)),prefix_growth_decimal=list(map(float,growth)),hypothetical_k10_if_factor30_each_future_step=prefix[-1]['regions']*30**5,noisy_rows=reports,noisy_ratio_ranges=ranges,raw_half_records=raw_count,new_available_exports=exports,review_cpu=time.process_time()-start,new_author_import_model_geometry_mass_MC_calls=0,
 findings=['Four prefix summaries reproduce the report and self-declare complete coverage, but no saved per-subtree history sets or exact feasibility certificates are available for independent cardinality verification. T219 independent recount remains required.','Prefix traversal stores selected gate words directly; noisy exact-run regions instead deduplicate gate words evaluated at rounded leaf centroids. Neither is an exact-rational feasibility certificate.','sec_cpu in prefix.py is front wall time plus summed worker time.time durations, not measured process CPU. Geometry operation counts are inherited partial proxies.','Finite ratios23.2941/26.5556/30.0682 do not prove any future lower growth bound. The15.367e12 projection assumes factor30 for each of five unseen steps.','Time-stopped cells_est is not generally unbiased: with two simultaneous subtrees of runtime1/100 and sizes1/1000, cap0.5 stops after the first completion and estimates2 instead of1001. Completek2..5 records do not use this estimator.','All18 noisy/clean rows and432 half records conserve saved outputs/counters. Region totals themselves are summary-only; pattern sets are not in the per-half logs.','Noise ranges describe16 fixed copies of H8T4/H16T4 zoo0, not the targetM16T10 or a distributional guarantee. Relative levels share fixed random streams and rescale perturbations.'],
 disposition='Saved arithmetic and source interpretation only; no confirmation of exact region counts, future enumeration lower bound, numerical truth or complete FLOP cost. Missing trained T5 arrays at olda5ab8757 is superseded by two tracked blobs addedde4b8350; provenance/schema review still required before any DRI-authorized numerical use.')
 (H/'t192_s4_saved_review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(complete=True,raw_half_records=raw_count,ranges=ranges,review_cpu=out['review_cpu'])))
if __name__=='__main__':main()
