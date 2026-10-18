"""Independent saved aggregate audit; no author imports, arrays or evaluator."""
from pathlib import Path
from fractions import Fraction as Q
import ast,hashlib,json,math,os,resource,time,traceback
HERE=Path(__file__).resolve().parent
OWNER=Path('/workspace/work/recover-small_h4_4/review/random_weight_evidence')
A=OWNER/'frozen/author/m16/approaches/random_control'
OUT=HERE/'output'
MODELS=['target']+[f'{s}{i}' for s in ['frob','row'] for i in range(5)]+['perm0','flip0']


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def exact(obj):return Q(obj['numerator'],obj['denominator'])
def check_exact(obj,q):
    assert exact(obj)==q
    assert obj['display']==float(q)
def middle(qs):
    a=sorted(qs);n=len(a)
    return (a[(n-1)//2]+a[n//2])/2


def main():
    assert not OUT.exists() or not any(OUT.iterdir())
    OUT.mkdir(exist_ok=True);start=time.process_time()
    report={'complete':False,'pass_all':False,'raw_rows':0,'author_calls':0,'array_decodes':0}
    def save():
        report['cpu_seconds']=time.process_time()-start
        tmp=OUT/'results.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');os.replace(tmp,OUT/'results.json')
    try:
        p=read(HERE/'pins.json');gate=read(HERE/'gate.json')
        required=[HERE/'check.py',HERE/'PROTOCOL.md',HERE/'pins.json',Path(p['source_gate'])]
        assert gate['ready'] and all(str(x) in gate['pins'] for x in required)
        for path,h in {**p['pins'],**gate['pins']}.items():assert sha(path)==h,path
        source=read(p['source_gate']);assert source['complete'] and source['pass_all'] and source['execution_authorized']
        assert str(HERE/'check.py') in source['pins'] and str(HERE/'PROTOCOL.md') in source['pins']
        for path,h in source['pins'].items():assert sha(path)==h,path
        mandatory=[OWNER/'saved/output/results.json',OWNER/'saved/output/progress.jsonl',OWNER/'saved/run1.log',OWNER/'saved/gate.json']
        assert all(str(x) in p['pins'] for x in mandatory)
        assert all(0<v<=20 for v in resource.getrlimit(resource.RLIMIT_CPU))
        assert all(0<v<=2*1024**3 for v in resource.getrlimit(resource.RLIMIT_AS))
        d=read(mandatory[0]);assert d['complete'] is True and d['pass_all'] is True
        assert d['author_calls']==d['array_decodes']==0
        raw={};by={(r['model'],r['k']):r for r in d['rows']}
        assert len(d['rows'])==len(by)==39
        for model in MODELS:
            for k in [2,3,4]:
                r=read(A/'out'/f'{model}_k{k}.json');v=by[model,k]
                assert v['status']=='complete' and v['raw']==r
                assert r['model']==model and r['k']==k
                assert all(type(r[x]) is int and r[x]>0 for x in ['cells','regions','ops_geometry'])
                assert r['regions']<=r['cells']
                assert all(math.isfinite(r[x]) and r[x]>=0 for x in ['cpu','cpu_total']) and r['cpu']<=r['cpu_total']
                if k==2:
                    assert all(math.isfinite(r[x]) and 0<=r[x]<=1 for x in ['acc_1e6','active_frac_T10'])
                    hits=Q(r['acc_1e6'])*1000000
                    candidates=[hits.numerator//hits.denominator,hits.numerator//hits.denominator+1]
                    possible=[n for n in candidates if n/1000000==r['acc_1e6']]
                    assert len(possible)==1 and possible[0]==v['reported_mean_compatible_hits']
                if k==4:
                    assert set(r['N'])=={'1e-02','1e-04','1e-06'}
                    ns=[r['N'][delta] for delta in ['1e-02','1e-04','1e-06']]
                    assert all(type(n) is int and 1<=n<=r['regions'] for n in ns) and ns==sorted(ns)
                    assert math.isfinite(r['mass_sum_minus_1']) and r['mass_sum_minus_1']>-1
                    assert math.isfinite(r['inv_simpson']) and r['inv_simpson']>0
                raw[model,k]=r;report['raw_rows']+=1;save()
        summary=read(A/'summary.json');assert set(summary)==set(MODELS)
        for m in MODELS:
            s=summary[m];a,b,c=[raw[m,k] for k in [2,3,4]]
            assert s==dict(regions=[a['regions'],b['regions'],c['regions']],growth=[b['regions']/a['regions'],c['regions']/b['regions']],acc=a['acc_1e6'],active=a['active_frac_T10'],N=c['N'],inv_simpson=c['inv_simpson'],msum=c['mass_sum_minus_1'])
        log={}
        for line in (A/'drive.log').read_text().splitlines():
            model,k,rest=line.split(' ',2);key=model,int(k);assert key not in log
            log[key]=ast.literal_eval(rest);assert log[key]=={key2:value for key2,value in raw[key].items() if key2!='model'}
        missing=[[m,k] for m in MODELS for k in [2,3,4] if (m,k) not in log]
        assert d['log_coverage']['matched']==len(log) and d['log_coverage']['missing']==missing
        table={}
        for line in (A/'README.md').read_text().splitlines():
            c=[v.strip() for v in line.split('|')]
            if len(c)==9 and c[1] in MODELS:assert c[1] not in table;table[c[1]]=c[2:-1]
        assert set(table)==set(MODELS)
        for m in MODELS:
            s=summary[m]
            expected=[' / '.join(format(x,',') for x in s['regions']),', '.join('x'+format(x,'.1f') for x in s['growth']),
                      ' / '.join(format(s['N'][k],',') for k in ['1e-02','1e-04','1e-06']),format(s['inv_simpson'],'.0f'),format(s['acc'],'.4f'),format(s['active'],'.2f')]
            assert table[m]==expected
        sets={'frob5':MODELS[1:6],'row5':MODELS[6:11],'random10':MODELS[1:11],'nontarget12_including_controls':MODELS[1:]}
        assert set(d['groups'])==set(sets)
        target=Q(raw['target',4]['regions'],raw['target',3]['regions'])
        for name,ms in sets.items():
            v=d['groups'][name];assert v['models']==ms and v['preregistered_decision_group']==(name=='random10')
            g3=middle([Q(raw[m,3]['regions'],raw[m,2]['regions']) for m in ms])
            g4=middle([Q(raw[m,4]['regions'],raw[m,3]['regions']) for m in ms]);relative=g4/target
            for field,q in [('median_growth_k3',g3),('median_growth_k4',g4),('relative_growth_k4',relative)]:check_exact(v[field],q)
            ratios_by_delta={}
            for delta in ['1e-02','1e-04','1e-06']:
                q=sorted(Q(raw[m,4]['N'][delta],raw['target',4]['N'][delta]) for m in ms);ratios_by_delta[delta]=q
                for field,value in [('minimum',q[0]),('median',middle(q)),('maximum',q[-1])]:check_exact(v['reported_N_ratio'][delta][field],value)
            strong=relative>=2 and middle(ratios_by_delta['1e-04'])>=3
            within=7*relative.denominator<=10*relative.numerator<=13*relative.denominator
            assert v['descriptive_category']==('strong' if strong else 'within30percent' if within else 'moderate')
        check_exact(d['reported_cpu_total_sum'],sum(Q(r['cpu_total']) for r in raw.values()))
        check_exact(d['max_abs_reported_mass_sum_minus_1'],max(abs(Q(raw[m,4]['mass_sum_minus_1'])) for m in MODELS))
        events=[json.loads(line) for line in mandatory[1].read_text().splitlines()]
        expected=[{'status':'started'}]+[{'model':m,'k':k,'status':s} for m in MODELS for k in [2,3,4] for s in ['started','complete']]+[{'status':'complete'}]
        assert events==expected and len(events)==80
        assert d['summary_models_checked']==d['rounded_table_rows_checked']==13
        assert math.isfinite(d['cpu_seconds_main']) and 0<=d['cpu_seconds_main']<16
        assert all(0<v<=20 for v in d['resource_limits']['CPU']) and all(0<v<=2*1024**3 for v in d['resource_limits']['AS'])
        text=mandatory[2].read_text();assert 'Traceback' not in text and 'real ' in text and 'user ' in text and 'sys ' in text
        report.update(summary_models=13,table_models=13,groups=4,events=80,log_rows=len(log),missing_log=missing,owner_cpu=d['cpu_seconds_main'],owner_resource_limits=d['resource_limits'],pins=p['pins'],qualifications=['Aggregate arithmetic only; no per-history masses, minimality or integration checked.','Compatible hits are inferred from reported means, not observed independent trials.','Conditional source/runtime/model identity remains unverified; no true training-effect or complete cost inference.'])
        assert time.process_time()-start<17
        report['complete']=report['pass_all']=True;save()
    except BaseException:
        report['complete']=report['pass_all']=False;report['failure']=traceback.format_exc();save();raise

if __name__=='__main__':main()
