from pathlib import Path
import numpy as np
import json,time
start=time.process_time();root=Path('/workspace/work/recover-small_h2_56/m16/approaches/face_surplus_review')
a=np.load(root/'data/T5_sigma0.0025_R23_axis2.npz');reference=json.loads((root/'results.json').read_text())['cases'][0]
rows=[];maxdifference=0.
for name,specfield,valfield in [('full_whole',None,'full_values'),('split_interior','panel_specs','panel_values'),('equal_count_interior','uniform_specs','uniform_values')]:
 rms=[];l1=[]
 for level in range(4,9):
  ids=np.arange(0,257,256//(2**level));left,middle,right=ids[:-2:2],ids[1:-1:2],ids[2::2]
  values=a[valfield];delta=values[:,middle]-(values[:,left]+values[:,right])/2
  if specfield is None:
   grid=a['full_grid'];valid=(grid[left]>=.05)&(grid[right]<=.95);delta=delta[:,valid]
   energy=(delta**2).mean(axis=1);absolute=abs(delta).mean(axis=1)
  else:
   specs=a[specfield];energy=np.zeros(32);absolute=np.zeros(32)
   for p,(line,lo,hi) in enumerate(specs):
    energy[int(line)]+=(hi-lo)/.9*np.mean(delta[p]**2)
    absolute[int(line)]+=(hi-lo)/.9*np.mean(abs(delta[p]))
  rms.append(float(np.sqrt(energy.mean())));l1.append(float(absolute.mean()))
  old=reference[name]['levels'][level-1];maxdifference=max(maxdifference,abs(rms[-1]-old['rms']),abs(l1[-1]-old['l1']))
 slope=float(-np.polyfit(np.arange(4,9),np.log2(rms),1)[0])
 rows.append(dict(name=name,rms=rms,l1=l1,slope=slope,author_slope=reference[name]['rms_exponent_4_8']))
 assert abs(slope-reference[name]['rms_exponent_4_8'])<1e-11
# Each line's exact x² split RMS factor sqrt(sum(relative panel widths**5)).
errors=[]
for line in range(32):
 specs=a['panel_specs'][a['panel_specs'][:,0]==line];ratios=(specs[:,2]-specs[:,1])/.9
 predicted=np.sqrt(np.sum(ratios**5));energy=0.
 for _,lo,hi in specs:
  x=np.linspace(lo,hi,257);delta=x[1:-1:2]**2-(x[:-2:2]**2+x[2::2]**2)/2
  energy+=(hi-lo)/.9*np.mean(delta**2)
 expected_unsplit=(.9/256)**2
 errors.append(abs(np.sqrt(energy)/expected_unsplit-predicted))
result=dict(rows=rows,max_statistic_difference=maxdifference,quadratic_panel_factor_max_error=max(errors),cpu_seconds=time.process_time()-start,new_face_evaluations=0)
Path(__file__).with_name('surplus_review_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
