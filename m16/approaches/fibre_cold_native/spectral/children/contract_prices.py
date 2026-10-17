"""Contract independently reviewed numeric subsets, without rerunning kernels."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
OWNER=Path('/workspace/work/recover-small_h2_56/m16/approaches/rank_numeric_envelope')
def main():
    dest=HERE/'price_contraction.json'
    assert not dest.exists()
    d=json.loads((HERE/'capture.json').read_text());assert d['complete']
    totals={k:0 for k in ['dnrm2','dscal','dlapy2','dlae2','dlasrt','dlanst']}
    rows=[]
    for row in d['rows']:
        prices=dict.fromkeys(totals,0)
        for event in row['events']:
            f=event['function'];a=event['args']
            if f=='dnrm2':
                assert a[0]>=1 and a[1]>0
                prices[f]+=1 if a[0]==1 else 2*a[0]+4
            elif f=='dscal':
                assert a[0]>=1 and a[1]>0
                prices[f]+=a[0]+3
            elif f=='dlapy2':prices[f]+=13
            elif f=='dlae2':prices[f]+=28
            elif f=='dlasrt':
                n=a[0];assert 1<=n<=16;prices[f]+=n*(n-1)//2
            elif f=='dlanst':
                n=a[0];assert 1<=n<=16;prices[f]+=8*n-6
        for f in totals:totals[f]+=prices[f]
        rows.append({'saved_row':row['saved_row'],'name':row['name'],'prices':prices})
    assert totals==dict(dnrm2=484,dscal=72,dlapy2=1183,dlae2=168,dlasrt=201,dlanst=316),totals
    files=[HERE/'capture.json',HERE/'HELPER_PRICES.md']
    for sub in ['norm_review','helper_review']:
        files += [OWNER/sub/name for name in ['DERIVATION.md','protocol.json','dispatch.json']]
    pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result={'complete':True,'rows':rows,'totals':totals,'numeric_subset':sum(totals.values()),'pins':pins,'reviews':['e61dffe3','300df19b','d90477c1','0cfdfa39'],'scope':'Fixed saved calls only; parent/other BLAS/utility/control/memory/runtime remain separate.'}
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(totals))
if __name__=='__main__':main()
