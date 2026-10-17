"""Independent coordinate-elimination hull and saved leaf-sum review."""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
from scipy.spatial import ConvexHull,QhullError
start=time.process_time();here=Path(__file__).parent;src=Path('/workspace/work/recover-small_h2_56/m16/approaches/leaf_triangulation_review');d=json.loads((src/'results.json').read_text());assert d['complete'];assert hashlib.sha256((src/'raw_leaves.npz').read_bytes()).hexdigest()==d['raw_archive_sha256'];a=np.load(src/'raw_leaves.npz');rows=[];fails=[]
for j,record in enumerate(d['leaves']):
 R=a[f'leaf{j}_R'];simp=a[f'leaf{j}_simplices'];anchor=a[f'leaf{j}_anchor'];dots=R@anchor;assert min(dots)>0
 assert all(any(np.array_equal(v,r) for r in R) for s in simp for v in s)
 # Eliminate the coordinate with largest normal coefficient; area scale is ||a||/|a_p|.
 pivot=int(np.argmax(abs(anchor)));coords=(R/dots[:,None])[:,[k for k in range(4) if k!=pivot]]
 try:volume=float(ConvexHull(coords).volume)*float(np.linalg.norm(anchor))/abs(float(anchor[pivot]))
 except QhullError as err:fails.append(dict(leaf=j,failure=str(err)));continue
 delta=volume-record['source_volume3'];rows.append(dict(leaf=j,parent=record['parent'],alternative_hull=volume,source_delta=delta,old_raw_delta=volume-record['raw']['volume']))
 assert abs(delta)<1e-10
for p in d['parents']:
 leaves=[r for r in d['leaves'] if r['parent']==p['index']];assert len(leaves)==p['leaves'] and sum(r['simplices'] for r in leaves)==p['simplices']
 for method in ('raw','conditioned'):
  assert math.fsum(r[method]['volume'] for r in leaves)==p[method]['hull_sum'];assert math.fsum(abs(r[method]['source_minus_hull']) for r in leaves)==p[method]['sum_absolute_difference']
assert len(rows)+len(fails)==337 and sum(r['simplices'] for r in d['leaves'])==1762
out=dict(leaves=337,simplices=1762,all_source_simplex_vertices_in_original_leaf=True,alternative_coordinate_hull_failures=fails,maximum_source_alternative_absolute_difference=max(abs(r['source_delta']) for r in rows),maximum_raw_alternative_absolute_difference=max(abs(r['old_raw_delta']) for r in rows),maximum_parent_partition_residual=max(abs(p[k]['relative_partition_residual']) for p in d['parents'] for k in ('raw','conditioned')),review_cpu=time.process_time()-start,scope='Independent coordinate elimination and saved aggregation, common Qhull backend. Floating checks, not exact geometry or no-compensating-overlap certificate.')
(here/'leaf_geometry_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
