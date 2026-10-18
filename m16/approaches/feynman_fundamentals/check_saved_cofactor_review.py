"""Independent saved-minor elimination/sign/ledger check; no native calls."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/cofactor3_guard/saved_minors')
a=json.loads((P/'results.json').read_text());m=a['manifest'];assert a['complete']
for path,h in m['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
def determinant(A):
    B=[list(map(F,row)) for row in A];out=F(1)
    for k in range(3):
        j=next((j for j in range(k,3) if B[j][k]),None)
        if j is None:return F(0)
        if j!=k:B[k],B[j]=B[j],B[k];out=-out
        v=B[k][k];out*=v
        for j in range(k+1,3):
            t=B[j][k]/v
            for z in range(k+1,3):B[j][z]-=t*B[k][z]
    return out
def rat(x):return F(*map(int,x))
expected=[(label,node,pair) for label in m['labels'] for node in m['nodes'] for pair in m['pair_indices']]
assert [(r['label'],r['node'],r['pair_position']) for r in a['rows']]==expected and len(expected)==64
archive=Path(next(p for p in m['pins'] if p.endswith('panel_nodes.npz')))
counts=Counter(); widths=[]; candidate_errors=[];source_errors=[]
with np.load(archive) as z:
    for row in a['rows']:
        label,node,pair=row['label'],row['node'],row['pair_position']
        i,j=m['pair_order'][pair];k,l=sorted(set(range(4))-{i,j});sign=1 if (k+l)%2==0 else -1
        Rt=z[f'{label}_Rt'][node]
        minor=[[float(Rt[r,c]) for c in range(4) if c!=l] for r in range(4) if r!=k]
        assert [[v.hex() for v in r] for r in minor]==row['minor_hex']
        q=determinant(minor); signed=sign*q
        assert q==rat(row['exact_unsigned']) and signed==rat(row['exact_signed'])
        assert (k,l,sign)==(row['deleted_row'],row['deleted_column'],row['cofactor_sign'])
        source=-float(z[f'{label}_atan2_arguments'][pair,0,node])
        assert source.hex()==row['source_signed_point']
        lo,hi=map(float.fromhex,row['interval']);point=float.fromhex(row['candidate_signed_point'])
        assert F(lo)<=signed<=F(hi) and row['enclosed']
        assert row['contains_zero']==(lo<=0<=hi)==False
        e=F(point)-signed;se=F(source)-signed;w=F(hi)-F(lo)
        assert e==rat(row['candidate_error_exact']) and se==rat(row['source_error_exact'])
        assert w==rat(row['interval_width_exact']) and w/abs(signed)==rat(row['relative_width'])
        assert abs(e/signed)==rat(row['candidate_relative_error']) and abs(se/signed)==rat(row['source_relative_error'])
        labelc='tie' if abs(e)==abs(se) else 'candidate_better' if abs(e)<abs(se) else 'source_better'
        assert labelc==row['absolute_error_comparison'];counts[labelc]+=1
        assert row['primitive_status']=='fast_enclosure'
        comparisons=1+sum(1 if x==0 else 3 for r in minor for x in r)
        expected_counts=dict(to_float=9,isfinite=9,abs=9,comparison=comparisons,float_mul=19,float_sub=5,float_add=7,nextafter=2)
        assert row['primitive_counts']==expected_counts
        assert row['signed_wrapper_negations']==(3 if sign<0 else 0)
        widths.append(w/abs(signed));candidate_errors.append(abs(e/signed));source_errors.append(abs(se/signed))
assert dict(counts)==a['summary']['point_comparisons']
for key,values in [('relative_width',widths),('candidate_relative_error',candidate_errors),('source_relative_error',source_errors)]:
    d=a['summary']['max_'+key];assert rat(d['value'])==max(values)==values[d['row']]
out=dict(complete=True,cases=64,all_exact_signed_enclosures=True,point_comparisons=dict(counts),
         max_relative_width=float(max(widths)),max_candidate_relative_error=float(max(candidate_errors)),
         max_source_relative_error=float(max(source_errors)),source_sha256=a['source_sha256'],
         new_native_determinant_mass_model_calls=0,
         scope='Fixed64 stored nonprincipal cofactors. Exact Gaussian elimination and signs/category counts independently checked. No universal point improvement, native speedup or propagated probability-error claim.')
(HERE/'saved_cofactor_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
