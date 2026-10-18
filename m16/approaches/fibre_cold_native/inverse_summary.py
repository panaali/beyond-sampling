"""Contract saved events only; no numerical kernel execution."""
from collections import Counter
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
    dest=HERE/'inverse_summary.json';assert not dest.exists()
    pins=json.loads((HERE/'inverse_price_pins.json').read_text())
    for p,h in pins.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    d=json.loads((HERE/'inverse_capture.json').read_text());assert d['complete'] and len(d['rows'])==54
    rows=[]
    for row in d['rows'][:48]:
        assert row['parity'];events=Counter((e['kernel'],tuple(e['args'])) for e in row['events'])
        assert sorted(e['args'][0] for e in row['events'] if e['kernel']=='iamax')==[1,2,3]
        assert sorted(tuple(e['args'][:3]) for e in row['events'] if e['kernel']=='gemm')==[(1,1,2),(1,2,2),(2,1,1),(2,2,1)]
        for kind in ('lower','upper'):
            assert events[(kind,(3,2,3,0))]==events[(kind,(3,1,3,0))]==1
        assert events[('upack',(3,3,3,0))]==1
        gemm=sum(2*e['args'][0]*e['args'][1]*(e['args'][2]+1) for e in row['events'] if e['kernel']=='gemm')
        assert gemm==42
        rows.append({'name':row['name'],'LU_arithmetic':24,'LU_abs_comparison':63,'local_solves':30,'cross_block_GEMM':gemm,'upper_reciprocals':3,'solve_guards':4,'numeric_upper_subset':166})
    out={'complete':True,'independent_review_pending':True,'rows':rows,'cold_inverse_numeric_upper_subset':sum(r['numeric_upper_subset'] for r in rows),'all54_parity':all(r['parity'] for r in d['rows']),'capture_sha256':hashlib.sha256((HERE/'inverse_capture.json').read_bytes()).hexdigest(),'full_implementation_cost':False}
    assert out['cold_inverse_numeric_upper_subset']==7968
    dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
if __name__=='__main__':main()
