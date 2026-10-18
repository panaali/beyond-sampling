"""Independent saved arithmetic audit; no author code imports or calls."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,math,time,resource
H=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rat(x):return F(int(x[0]),int(x[1]))
def rounded(x):
    # Independently round a positive rational to nearest-even normal binary64.
    assert x>0
    n,d=x.numerator,x.denominator
    e=n.bit_length()-d.bit_length()
    if x < (F(2)**e):e-=1
    shift=52-e
    if shift>=0:n<<=shift
    else:d<<=-shift
    q,r=divmod(n,d)
    if 2*r>d or (2*r==d and q%2):q+=1
    return F(q)*(F(2)**(-shift))
def main():
    target=H/'saved_review.json';attempt=H/'saved_review_attempt.json'
    assert not target.exists() and not attempt.exists()
    start=time.process_time();p=json.loads((H/'saved_protocol.json').read_text())
    attempt.write_text(json.dumps({'started':True,'pins':p['pins']})+'\n')
    out={'complete':False,'assignment':'0a43c9f8','author_raw':'54502cb1','new_author_driver_estimator_Q_RNG_calls':0}
    try:
        for path,sha in p['pins'].items():assert digest(path)==sha,path
        root=Path(p['author']);proto=json.loads((root/'run_protocol.json').read_text())
        for path,sha in proto['pins'].items():assert digest(path)==sha,path
        raw=json.loads((root/'results.json').read_text());assert raw['complete']
        assert raw['protocol_sha256']==digest(root/'run_protocol.json')
        alloc=json.loads(Path(proto['allocation']).read_text())['plain']
        const=raw['constants'];c=rounded(F(1.6)+F(.1));factor=rounded(F(3.3)/F(1.6))
        assert F(float.fromhex(const['c_hex']))==c and F(float.fromhex(const['factor_hex']))==factor
        assert const['allocation_hex']==[a.hex() for a in alloc]
        assert const['cost_q_hex']==(1.6).hex() and const['cost_draw_hex']==(.1).hex()
        assert (const['batch'],const['fpe_flops'],rat(const['rule5_extra']))==(50000,6000,F(1,25))
        progress=[json.loads(x) for x in (root/'progress.jsonl').read_text().splitlines()]
        expected=[{'stage':'phase_attempt','protocol_sha256':digest(root/'run_protocol.json')}]
        assert len(raw['controls'])==4 and len(raw['rows'])==6
        for row,m,v in zip(raw['controls'],[1,50000,50001,100000],[50000,50000,100000,100000],strict=True):
            assert row==dict(M=m,expected=v,value=v,passed=True)
            assert (v-50000)<m<=v and v%50000==0
            expected += [dict(stage='control_attempt',M=m),dict(stage='control_result',row=row)]
        checked=[]
        for row,budget in zip(raw['rows'],[1000,4000,16000,64000,256000,1000000],strict=True):
            assert time.process_time()-start<17
            assert row['budget']==budget
            n=(F(budget)/c).__floor__();assert n==int(budget//float(c))==row['n_source']==row['n_exact_binary_c_floor']
            products=[rounded(F(a)*n) for a in alloc]
            assert products==[F(float.fromhex(x)) for x in row['product_hex']]
            quotas=[max(4,v.__floor__()) for v in products];m=sum(quotas)
            assert quotas==row['quotas'] and m==row['M']
            d=row['D_min'];assert d%50000==0 and d-50000<m<=d
            product=rounded(m*c);charge=product.__ceil__();reported=rounded(charge*factor)
            assert product==F(float.fromhex(row['raw_product_hex'])) and charge==row['raw_charge']
            assert reported==rat(row['reported_exact'])==F(float.fromhex(row['reported_hex']))
            # Derive dense floor from dimensions instead of copying owner7040/6820 constants.
            dense_flops=10*m*16*16*2+3*2*m*16*10*2
            classic_units=10*m*16*31+3*2*m*10*31
            assert dense_flops==row['canonical_dense_flops'] and classic_units==row['classic_dot_units']
            dense=F(dense_flops,6000);classic=F(classic_units,6000);extra=F(d-m,25)
            checks={'canonical_dense_FPE':dense,'classic_dot_FPE':classic,'extra_candidate_FPE':extra,'conditional_canonical_FPE':dense+extra,'conditional_classic_FPE':classic+extra,'conditional_gap':dense+extra-reported}
            for key,value in checks.items():assert rat(row[key])==value,key
            assert row['dense_only_status']==('floor_exceeds_report' if dense>reported else 'inconclusive')
            assert row['conditional_status']==('conditional_undercharge' if dense+extra>reported else 'inconclusive')
            assert row['classic_conditional_status']==('conditional_undercharge' if classic+extra>reported else 'inconclusive')
            display={'reported':reported,'canonical_dense':dense,'extra':extra,'conditional':dense+extra,'gap':dense+extra-reported}
            assert row['displayed']=={k:float(v) for k,v in display.items()}
            expected +=[dict(stage='budget_attempt',budget=budget),dict(stage='budget_result',row=row)]
            checked.append(dict(budget=budget,M=m,D_min=d,status=row['conditional_status'],gap=str(dense+extra-reported)))
        expected.append(dict(stage='complete'));assert progress==expected and len(progress)==22
        assert raw['new_author_estimator_model_RNG_native_reference_calls']==0
        out.update(complete=True,verified_pins=len(p['pins'])+len(proto['pins']),rows=checked,controls=4,progress=22,conditional_undercharge_budgets=[r['budget'] for r in checked if r['status']=='conditional_undercharge'],limitations='Stated dense convention and finite distinct-top2 premise; candidate allowance depends on rule5 prefix interpretation. No actual D/history/fullnative/MSE claim.')
    except BaseException as exc:
        out['failure']=repr(exc);raise
    finally:
        out.update(cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        target.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))
if __name__=='__main__':main()
