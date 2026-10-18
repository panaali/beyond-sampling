"""Independent saved-only T11589 review: positive angular integral, static ledgers."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math,time
import mpmath as mp
HERE=Path(__file__).resolve().parent

def main():
    out=HERE/'inner5_review.json';progress=HERE/'inner5_review_progress.jsonl'
    assert not out.exists() and not progress.exists()
    protocol=json.loads((HERE/'inner5_review_protocol.json').read_text())
    for p,pin in protocol['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==pin,p
    data=json.loads(Path(protocol['results']).read_text());assert data['complete'] and len(data['rows'])==26
    mp.mp.dps=100;start=time.process_time();rows=[];maxerr=mp.mpf(0);finite=0
    def mf(x):return mp.mpf(x.numerator)/x.denominator
    for old in data['rows']:
        with progress.open('a') as f:f.write(json.dumps(dict(stage='attempt',id=old['id']))+'\n')
        pf=[float.fromhex(x) for x in old['P']];df=[float.fromhex(x) for x in old['D']]
        independent=None
        if not all(math.isfinite(x) for x in pf+df):domain='nonfinite'
        else:
            p=list(map(F.from_float,pf));d=list(map(F.from_float,df))
            a=sum(x*x for x in d);b=sum(x*y for x,y in zip(p,d));c=sum(x*x for x in p);delta=a*c-b*b
            assert delta>=0
            if a==0:
                domain='singular' if c==0 else 'finite_positive'
                if c:independent=mf(c)**mp.mpf('-2.5')
            elif delta==0:
                domain='singular' if c==0 or 1+b/c<=0 else 'finite_positive'
                if domain=='finite_positive':
                    # Direct positive z-integral, not the author's collinear formula.
                    independent=mp.quad(lambda z:(mf(c)+2*mf(b)*z+mf(a)*z*z)**mp.mpf('-2.5'),[0,1],maxdegree=5)
            else:
                domain='finite_positive';sd=mp.sqrt(mf(delta))
                t0=mp.atan(mf(b)/sd);t1=mp.atan(mf(a+b)/sd)
                # Completing the square and z -> atan gives a positive cos^3
                # integral, avoiding subtraction of the author's antiderivative.
                integral=mp.quad(lambda theta:mp.cos(theta)**3,[t0,t1],maxdegree=5)
                independent=mf(a)**mp.mpf('1.5')/mf(delta)**2*integral
        assert domain==old['reference240']['domain']==old['reference280']['domain']
        relative=None;candidate_error=None
        if independent is not None:
            finite+=1;relative=abs(independent-mp.mpf(old['reference280']['value']))/independent
            assert relative<mp.mpf('1e-50'),(old['id'],relative)
            maxerr=max(maxerr,relative)
            assert old['reference_stable']
            if old['candidate']['status']=='ok':
                candidate_error=abs(mp.mpf(old['candidate']['value'])-independent)/independent
                assert abs(candidate_error-mp.mpf(old['relative_error']))<mp.mpf('1e-45')
        status=old['candidate']['status'];branch=old['candidate']['branch']
        if status=='nonfinite_input':expected=(0,10,0,0,10)
        elif status=='origin':expected=(0,29,10,0,10)
        elif status=='collinear_or_underflow':expected=(86,32,10,0,10)
        elif branch=='constant':expected=(35,32,10,1,11)
        elif branch=='same_sign':expected=(114,41,12,3,11)
        elif branch=='cross_or_endpoint':expected=(110,39 if status=='arithmetic_failure' else 41,12,3,10 if status=='arithmetic_failure' else 11)
        else:raise AssertionError(('unreviewed branch',old['id'],status,branch))
        actual=tuple(old['candidate']['counts'][k] for k in ['arithmetic','comparison_boolean','abs','sqrt','finite_check'])
        assert actual==expected,(old['id'],actual,expected)
        row=dict(id=old['id'],domain=domain,status=status,reference_relative_difference=mp.nstr(relative,70) if relative is not None else None,candidate_relative_error=mp.nstr(candidate_error,60) if candidate_error is not None else None,ledger=actual,ledger_pass=True)
        rows.append(row)
        with progress.open('a') as f:f.write(json.dumps(row)+'\n')
    result=dict(cases=len(rows),finite_reference_cases=finite,rows=rows,max_reference_relative_difference=mp.nstr(maxerr,70),cpu_seconds=time.process_time()-start,candidate_model_mass_outer_rule_calls=0,scope='Independent exact domain and positive angular numerical integral audit, not a rigorous quadrature certificate; static source ledgers exclude native/control/runtime. No author candidate rerun.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['cases','finite_reference_cases','max_reference_relative_difference','cpu_seconds']}))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        with (HERE/'inner5_review_failures.jsonl').open('a') as f:f.write(json.dumps(dict(type=type(e).__name__,message=str(e)))+'\n')
        raise
