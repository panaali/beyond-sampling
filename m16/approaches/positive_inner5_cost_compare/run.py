"""Fixed saved-only declared expression model, no numerical source execution."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,json,time,ast
H=Path(__file__).resolve().parent
CATS=['arithmetic','comparison_boolean','abs','sqrt','finite_check','power']
def old_vector(n):return dict(arithmetic=51*n**4+3,comparison_boolean=1,abs=1,sqrt=0,finite_check=0,power=6*n**4+1)
def pack(x):
    if isinstance(x,F):return str(x)
    if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [pack(v) for v in x]
    return x

def main():
    out=H/'results.json';progress=H/'progress.jsonl';assert not out.exists() and not progress.exists()
    P=json.loads((H/'protocol.json').read_text());R=json.loads((H/'run_protocol.json').read_text())
    for pins in [P['pins'],R['pins']]:
        for p,h in pins.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    old=json.loads(Path(P['old_results']).read_text());new=json.loads(Path(P['new_results']).read_text());assert old['complete'] and new['complete']
    source=Path(R['old_source']).read_text();tree=ast.parse(source);body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['set_rules5','_orth5']]
    assert hashlib.sha256(ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False).encode()).hexdigest()==old['extracted_ast_sha256']
    hand=json.loads(Path(P['old_hand_counts']).read_text())['thin'];M=6**4
    assert hand=={'numeric_expression:Mult':3,'numeric_expression:Div':M+1,'numeric_comparison:Lt':1,'opaque_matmul:MatMult':5*M,'opaque_power:Pow':6*M+1,'opaque_call:np.linalg.det':1,'opaque_call:abs':1,'opaque_call:np.sum':M+1}
    def record(r):
        with progress.open('a') as f:f.write(json.dumps(pack(r),allow_nan=False)+'\n')
    start=time.process_time();record(dict(stage='start'));rows=[];old_by={(r['fixture'],r['side'],r['order']):r for r in old['rows']};newrows=[r for r in new['rows'] if r['kind']=='split']
    refs={(r['fixture'],r['side']):r['reference120'] for r in old['children']};assert len(old_by)==len(newrows)==66 and len(refs)==22
    groups={}
    for r in newrows:
        key=(r['fixture'],r['side'],r['order']);o=old_by[key];n=r['order'];assert n in [6,10,16]
        record(dict(stage='child_attempt',fixture=key[0],side=key[1],order=n))
        assert r['complete'] and r['finite_positive'] and o['finite'] and o['probability_range'] and 'exception' not in o
        assert o['source_simplices']==1 and o['source_proxy_ops']==40*n**4+125
        assert r['reference']==refs[key[:2]] and r['value'].hex()==r['value_hex'] and o['value'].hex()==o['returned_hex']
        assert set(r['branches'])<=set(['cross_or_endpoint','same_sign']) and sum(r['branches'].values())==n**3
        vec={k:r['inner_counts'].get(k,0)+r['construction_assembly_counts'].get(k,0) for k in CATS}
        total=sum(vec.values());cross=r['branches'].get('cross_or_endpoint',0);same=r['branches'].get('same_sign',0)
        assert total==239*cross+243*same+9 and r['opaque_native_calls']=={'det_float64_5x5':1}
        ov=old_vector(n);assert sum(ov.values())==57*n**4+6
        errorold=F.from_float(o['value'])-F(r['reference']);errornew=F.from_float(r['value'])-F(r['reference'])
        row=dict(fixture=key[0],side=key[1],order=n,old_vector=ov,new_vector=vec,old_unit_opaque_total=sum(ov.values()),new_unit_opaque_total=total,old_legacy_proxy=o['source_proxy_ops'],opaque_det5_calls_each=1,old_value_hex=o['returned_hex'],new_value_hex=r['value_hex'],old_exact_reference_error=errorold,new_exact_reference_error=errornew,source_observer_scope='forced-thin old branch, diagnostic contraction expansion; actual backend instructions unpriced')
        rows.append(row);groups.setdefault((key[0],n),[]).append((row,o,r));record(row)
    gr=[];summary=[]
    for (fixture,n),members in groups.items():
        assert sorted(r[0]['side'] for r in members)==[0,1]
        priorold=next(g for g in old['groups'] if g['fixture']==fixture and g['order']==n)
        priornew=next(g for g in new['groups'] if g['fixture']==fixture and g['order']==n and g['kind']=='split')
        ref=F(priornew['reference']);vo=sum((F.from_float(r[1]['value']) for r in members),F(0));vn=sum((F.from_float(r[2]['value']) for r in members),F(0));eo=vo-ref;en=vn-ref
        absold=abs(eo)<=F(1,10**10);absnew=abs(en)<=F(1,10**10);relold=abs(eo/ref)<=F(1,10**6);relnew=abs(en/ref)<=F(1,10**6)
        assert (absold,relold)==(priorold['absolute_pass'],priorold['relative_pass']) and (absnew,relnew)==(priornew['absolute_pass'],priornew['relative_pass'])
        a=Counter();b=Counter()
        for row,_,_ in members:a.update(row['old_vector']);b.update(row['new_vector'])
        g=dict(fixture=fixture,order=n,old_vector=dict(a),new_vector=dict(b),old_unit_opaque_total=sum(a.values()),new_unit_opaque_total=sum(b.values()),ratio=F(sum(b.values()),sum(a.values())),old_absolute_pass=absold,new_absolute_pass=absnew,old_relative_pass=relold,new_relative_pass=relnew,old_exact_error=eo,new_exact_error=en,opaque_det5_calls_each=2,diagnostic_parent_addition_each=1)
        gr.append(g);record(dict(stage='parent',row=g))
    for n in [6,10,16]:
        gs=[g for g in gr if g['order']==n];assert len(gs)==11
        a=Counter();b=Counter()
        for g in gs:a.update(g['old_vector']);b.update(g['new_vector'])
        M=n**4;K=128
        setupold=dict(arithmetic=25*M+3*n+5*K,power=2*M,negation=K,exp=K,opaque_leggauss_calls=2)
        summary.append(dict(order=n,old_vector=dict(a),new_vector=dict(b),old_unit_opaque_total=sum(a.values()),new_unit_opaque_total=sum(b.values()),ratio=F(sum(b.values()),sum(a.values())),old_legacy_proxy=sum(r['old_legacy_proxy'] for r in rows if r['order']==n),old_absolute_pass=sum(g['old_absolute_pass'] for g in gs),new_absolute_pass=sum(g['new_absolute_pass'] for g in gs),old_relative_pass=sum(g['old_relative_pass'] for g in gs),new_relative_pass=sum(g['new_relative_pass'] for g in gs),opaque_det5_calls_each=22,parent_assembly_additions_each=11,old_universal_setup=setupold,new_universal_mapping=3*n,new_executed_staged_mapping=6*n,new_required_source_generation_calls=1,setup_scope='Model-free advice allowed under rule4 if declared. Native generators and array/memory/control work not priced; source validation references separate.'))
    assert new['canonical_mapping_named_arithmetic']==96 and new['actually_executed_staged_mapping_named_arithmetic']==192
    result=dict(complete=True,rows=rows,groups=gr,summary=summary,cpu_seconds=time.process_time()-start,scope=P['scope'],mass_model_geometry_reference_calls=0,convention='Diagnostic arithmetic expansion and unit opaque-call total only; no native/hardware upper/lower, runtime speed or sampler-MSE conclusion. Full library/count convention unresolved.')
    out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');print(json.dumps(dict(rows=len(rows),groups=len(gr),cpu_seconds=result['cpu_seconds'])))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        with (H/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(type=type(e).__name__,message=str(e)))+'\n')
        raise
