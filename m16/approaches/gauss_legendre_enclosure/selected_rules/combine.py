"""Fixed11 selected ideal-tail plus constant bounds; no quadrature values."""
from pathlib import Path
from fractions import Fraction as F
import sys,json,hashlib,time,importlib.util
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'perturbation'))
from bound import bound,pack
spec=importlib.util.spec_from_file_location('mapped_constants_frozen',H.parent/'perturbation'/'run.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
def rat(v):return F(int(v[0]),int(v[1]))
def main():
    out=H/'combined.json';progress=H/'combined_progress.jsonl';assert not out.exists() and not progress.exists()
    P=json.loads((H/'combine_protocol.json').read_text());assert not P['node_review'].startswith('pending')
    for p,h in P['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    selected=json.loads(Path(P['selector']).read_text());assert selected['complete']
    old=json.loads(Path(P['old_constants']).read_text());split=json.loads(Path(P['split_bounds']).read_text());assert old['complete'] and split['complete']
    def record(r):
        with progress.open('a') as f:f.write(json.dumps(pack(r),allow_nan=False)+'\n')
    start=time.process_time();record(dict(stage='constants_attempt'))
    # n6 mapping and exact bound rows are reused, never regenerated.
    constants={6:next(c for c in old['constants'] if c['order']==6)}
    fresh=[]
    for n in [12,22,28,30]:
        row=json.loads((H/f'nodes_{n}.json').read_text());assert row['complete'] and row['order']['bounded']
        cn=C.constants(row['order']);fresh.append(cn);constants[n]=pack(cn)
    record(dict(stage='constants_complete',new=fresh,reused_order=6))
    chosen=[g for g in selected['groups'] if g['panel']=='child'];assert len(chosen)==11
    assert {g['fixture']:g['selection']['selected_order'] for g in chosen}==P['fixed_selection']
    setups={(r['fixture'],r['side']):r for r in split['exact_setup']};rows=[];groups=[];reused=0;new=0
    for group in chosen:
        fixture=group['fixture'];sel=group['selection'];n=sel['selected_order'];cn=constants[n]
        assert sel['status']=='certified_ideal' and sel['steps'][-1]['order']==n and sel['steps'][-1]['qualifies']
        ideal=F(sel['bound_sum']);assert ideal==F(sel['steps'][-1]['bound_sum']) and F(0)<=ideal<=F(1,10**10)
        record(dict(stage='parent_attempt',fixture=fixture,order=n));children=[]
        for side in [0,1]:
            setup=setups[(fixture,side)]
            if n==6:
                prior=next(r for r in split['rows'] if r['fixture']==fixture and r['side']==side and r['order']==6)
                got=prior['result'];reused+=1;origin='reused'
            else:
                got=pack(bound(rat(setup['a']),rat(setup['M2']),rat(setup['det']),rat(cn['epsilon']),rat(cn['source_weight_sum']),rat(cn['delta_L1'])));new+=1;origin='new'
            row=dict(fixture=fixture,side=side,order=n,result=got,origin=origin);rows.append(row);children.append(row);record(row)
        good=all(c['result']['status']=='bounded' for c in children)
        const=sum((rat(c['result']['probability_upper']) for c in children),F(0)) if good else None
        total=ideal+const if good else None
        g=dict(fixture=fixture,order=n,status='bounded' if good else 'unknown',ideal_upper=ideal,constant_upper=const,combined_upper=total,below_1e_10=good and total<=F(1,10**10))
        groups.append(g);record(dict(stage='parent_complete',row=g))
    assert len(rows)==22 and reused==4 and new==18
    result=dict(complete=True,rows=rows,groups=groups,new_constants=fresh,new_child_bounds=new,reused_child_bounds=reused,source_mapping_arithmetic=sum(c['source_mapping_arithmetic'] for c in fresh),new_certificate_determinant_calls=0,new_source_leggauss_calls=0,exact_parent_assembly_additions_allowance=33,cpu_seconds=time.process_time()-start,scope=P['scope'],node_review=P['node_review'],inherited_setup='All universal GL construction, source-generation work, child exact certificate construction and selector advice remain in their original artifacts; reused does not mean zero production cost. Current exact mapping/bound categories are not native float cost.',tensor_mass_model_geometry_MC_calls=0)
    out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');print(json.dumps(dict(parents=len(groups),passed=sum(g['below_1e_10'] for g in groups),new_bounds=new,reused_bounds=reused,cpu_seconds=result['cpu_seconds'])))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        with (H/'combine_failures.jsonl').open('a') as f:f.write(json.dumps(dict(type=type(e).__name__,message=str(e)))+'\n')
        raise
