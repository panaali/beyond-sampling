"""Fixed T257 saved-JSON consistency only; never import author/model code."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal as D, localcontext
import hashlib,json,math,os,re,resource,time,traceback
P=Path(__file__).resolve().parent
BASE=P.parent
OUT=P/'output'
START=time.process_time()
R={'complete':False,'pass_all':False}
OWN=False

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def enc(q):return [str(q.numerator),str(q.denominator)]
def save():
    with (OUT/'results.tmp').open('w') as f:
        json.dump(R,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(OUT/'results.tmp',OUT/'results.json')
def event(kind,**kw):
    with (OUT/'progress.jsonl').open('a') as f:
        f.write(json.dumps(dict(kind=kind,cpu=time.process_time()-START,**kw),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    assert time.process_time()-START<17

def finite(x):return type(x) in (int,float) and math.isfinite(x)
def number(x):assert finite(x);return F(x)
def main():
    global OWN
    assert not OUT.exists(),'immutable output exists'
    OUT.mkdir();OWN=True;event('start')
    protocol=read(P/'protocol.json');gate=read(P/'gate.json')
    assert gate['execution_allowed'] is True
    for pins in [protocol['pins'],gate['pins']]:
        for p,h in pins.items():assert sha(p)==h,p
    req=[P/'check.py',P/'protocol.json',P/'PROTOCOL.md']
    assert all(gate['pins'][str(p)]==sha(p) for p in req)
    assert set(gate['reviews'])=={'prior_written','source_resource'}
    for role,v in gate['reviews'].items():
        assert all(re.fullmatch('[0-9a-f]{7,40}',v[k]) for k in ['commit','publication'])
        assert sha(v['path'])==v['sha256']==gate['pins'][v['path']]
        d=read(v['path']);assert d['complete'] is True and d['pass_all'] is True
        if role=='source_resource':
            assert d['execution_allowed'] is True
            assert all(d['pins'][str(p)]==sha(p) for p in req)
        else:
            for p in [BASE/'SOURCE_REVIEW.md',BASE/'manifest.json']:assert d['pins'][str(p)]==sha(p)
    R.update(gate_sha256=sha(P/'gate.json'),protocol_sha256=sha(P/'protocol.json'),limits={'cpu':list(resource.getrlimit(resource.RLIMIT_CPU)),'as':list(resource.getrlimit(resource.RLIMIT_AS))})
    save();assert R['limits']=={'cpu':[20,20],'as':[2*1024**3]*2}
    event('pins_checked')
    a=read(BASE/'snapshot/results/a30.json');z=read(BASE/'snapshot/results/zoo_code_seed0.json')
    assert set(a)=={'config','n_train','n_steps','wall_sec','curve','eval'}
    assert set(z)=={'seed','wall_sec','eval'} and type(z['seed']) is int and z['seed']==0
    cfg=a['config'];assert cfg['tag']=='a30' and cfg['H']==16 and cfg['T']==10 and cfg['seeds']==list(range(5))
    assert cfg['log2n']==30 and cfg['eval_n']==10**8
    assert type(a['n_train']) is int and type(a['n_steps']) is int
    assert a['n_train']==2**30 and a['n_steps']==a['n_train']//1024
    assert finite(a['wall_sec']) and a['wall_sec']>=0 and finite(z['wall_sec']) and z['wall_sec']>=0
    assert len(a['curve'])==64 and set(a['eval'])==set(map(str,range(5)))
    curve=[]
    for i,v in enumerate(a['curve']):
        assert set(v)=={'step','sec','loss','acc'} and type(v['step']) is int and v['step']==(i+1)*2**14
        assert finite(v['sec']) and v['sec']>=0 and len(v['loss'])==len(v['acc'])==5
        assert all(finite(x) and x>=0 for x in v['loss'])
        assert all(finite(x) and 0<=x<=1 for x in v['acc'])
        curve.append(v)
    assert curve[-1]['step']==a['n_steps']
    R['training_ledger']={'config':cfg,'n_train_per_model_reported':a['n_train'],'n_steps_reported':a['n_steps'],'curve_records':64,'model_slots_per_record':5,'loss_acc_pairs':320,'all_curve_records':curve,'reported_wall_seconds':{'a30':a['wall_sec'],'zoo_code_seed0':z['wall_sec']},'curve_seconds_nondecreasing':all(x['sec']<=y['sec'] for x,y in zip(curve,curve[1:])),'last_curve_seconds_not_above_reported_wall':curve[-1]['sec']<=a['wall_sec'],'historical_no_resume_not_verified':True}
    save();event('training_ledger_checked',records=64,model_slots=320)
    rows={'zoo_code_seed0':z['eval'],**{'a30_seed'+k:v for k,v in a['eval'].items()}}
    assert len(rows)==6;R['evaluations']={}
    expected={'n','seed','acc','se','blind_share','err_blind','err_23','err_12','err_other','blind_error_rate'}
    for key,v in rows.items():
        assert set(v)==expected and type(v['n']) is int and v['n']==10**8 and type(v['seed']) is int and v['seed']==777
        assert all(finite(v[k]) and 0<=v[k]<=1 for k in expected-{'n','seed'}) and v['blind_share']>0
        p=F(v['acc']);partition=p+sum((F(v[k]) for k in ['err_blind','err_23','err_12','err_other']),F(0))-1
        blind=F(v['err_blind'])/F(v['blind_share']);var=p*(1-p)/v['n']
        with localcontext() as ctx:
            ctx.prec=100
            se=(D(var.numerator)/D(var.denominator)).sqrt()
            se_diff=abs(D.from_float(v['se'])-se)
            se_allowance=D.from_float(8*math.ulp(float(se)))
        d={'stored':v,'partition_residual_exact':enc(partition),'blind_error_rate_from_stored_proportions_exact':enc(blind),'nominal_variance_from_stored_accuracy_exact':enc(var),'nominal_se_decimal100':str(se),'se_absolute_difference_decimal':str(se_diff),'se_allowance_decimal':str(se_allowance),'missing_integer_counts_not_reconstructed':True}
        R['evaluations'][key]=d;save()
        assert abs(partition)<=F(1,2**50)
        assert F(v['err_blind'])<=F(v['blind_share'])
        assert abs(F(v['blind_error_rate'])-blind)<=F(1,2**48)
        assert se_diff<=se_allowance
    event('evaluation_identities_checked',records=6)
    text=(BASE/'snapshot/README.md').read_text();table=[]
    for line in text.splitlines():
        if line.startswith('| zoo code unchanged'):
            key='zoo_code_seed0'
        elif line.startswith('| trainer.py seed '):
            m=re.match(r'\| trainer.py seed (\d)',line);assert m;key='a30_seed'+m[1]
        else:continue
        cells=[x.strip() for x in line.strip().strip('|').split('|')];assert len(cells)==6
        table.append((key,cells[1:]))
    assert len(table)==6 and {k for k,_ in table}==set(rows)
    R['table_comparisons']=[]
    for key,cells in table:
        for field,printed in zip(['acc','err_blind','err_23','err_12','err_other'],cells,strict=True):
            if not printed:continue
            assert re.fullmatch(r'0?\.\d+',printed)
            digits=len(printed.split('.')[1]);shown=F(printed);stored=F(rows[key][field])
            allowance=F(1,2*10**digits)+F(math.ulp(rows[key][field]))
            d={'record':key,'field':field,'printed':printed,'stored_exact':enc(stored),'absolute_difference_exact':enc(abs(shown-stored)),'rounding_allowance_exact':enc(allowance),'pass':abs(shown-stored)<=allowance}
            R['table_comparisons'].append(d);save();assert d['pass']
    matches=re.findall(r'zoo seed spread \((0\.\d+)-(0\.\d+);',text)
    assert matches and len(set(matches))==1
    lo,hi=map(F,matches[0]);assert lo<hi
    R['literal_printed_spread_diagnostic']={'printed_endpoints':list(matches[0]),'rows_inside_literal_closed_interval':{key:lo<=F(v['acc'])<=hi for key,v in rows.items()},'rounded_endpoints_do_not_define_exact_acceptance':True,'not_a_trajectory_distribution_or_runtime_equivalence_test':True}
    save();event('readme_transcription_checked',comparisons=len(R['table_comparisons']))
    R.update(complete=True,pass_all=True,author_model_rng_weight_calls=0,integer_paired_counts_reconstructed=0,cpu_main=time.process_time()-START)
    event('complete');save()

if __name__=='__main__':
    try:main()
    except BaseException as e:
        R.update(complete=False,pass_all=False,fatal={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
        if OWN:
            save()
            with (OUT/'progress.jsonl').open('a') as f:
                f.write(json.dumps({'kind':'fatal','type':type(e).__name__,'message':str(e)})+'\n');f.flush();os.fsync(f.fileno())
        raise
