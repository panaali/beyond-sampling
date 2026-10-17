"""Check saved primal/dual identities with Fraction, no author solver imports."""
from fractions import Fraction as F
from pathlib import Path
import gzip,json,time
start=time.process_time()
p=Path('/workspace/work/recover-small_h2_910/m16/approaches/bulk_error_mechanism')
data=json.loads((p/'readout.json').read_text())
g=Path('/workspace/clean-neurons/m16/approaches/small_ladder_review/rational_geometry/geometry_h4_t4.json.gz')
model=json.loads(gzip.decompress(g.read_bytes()))['model']
trained=[[F(int(a),int(b)) for a,b in row] for row in model[2]]
controls={'zero':[[F(0)]*4 for _ in range(4)],
          'duplicate':[[F(i+1) for i in range(4)] for _ in range(4)],
          'dominated':[[F(1-i)]*4 for i in range(4)],
          'identity':[[F(i==j) for i in range(4)] for j in range(4)]}
checked=[]
for name,w,rows in [('trained',trained,data['trained'])]+[(r['name'],controls[r['name']],r['rows']) for r in data['controls']]:
    H=len(w[0]);T=len(w)
    for row in rows:
        j=row['label'];h=list(map(F,row['nonzero_h_witness']));eps=F(row['optimum'])
        assert min(h)>=0 and sum(h)==1
        A=[[F(k==i) for k in range(H)]+[F(0)] for i in range(H)]
        for o in range(T):
            if o!=j:A.append([w[j][k]-w[o][k] for k in range(H)]+[F(-int(o<j or j==0))])
        z=h+[eps];slacks=[sum(a*b for a,b in zip(r,z)) for r in A]
        assert min(slacks)>=0
        mu=list(map(F,row['dual_multipliers']));bound=F(row['dual_bound'])
        assert min(mu)>=0 and bound==eps
        eq=[F(1)]*H+[F(0)]
        assert [bound*eq[k]-sum(mu[i]*A[i][k] for i in range(len(A))) for k in range(H+1)]==[F(0)]*H+[F(1)]
        scores=[sum(a*b for a,b in zip(r,h)) for r in w]
        winner=max(range(T),key=lambda o:scores[o])
        possible=j==0 or eps>0
        assert possible==row['possible_in_nonnegative_superset']
        if j>0 and possible:assert winner==j
        checked.append({'model':name,'label':j,'optimum':str(eps),'positive_witness':eps>0,'origin_exception':j==0})
out={'checked':checked,'count':len(checked),'cpu_seconds':time.process_time()-start,
     'scope':'nonnegative hidden superset only; no recurrence-reachability conclusion'}
Path(__file__).with_name('readout_lp_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
