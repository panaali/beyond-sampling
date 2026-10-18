"""Frozen31 synthetic controls; execution requires separately published gate."""
from pathlib import Path
from fractions import Fraction as F
from itertools import permutations, product
import hashlib, json, os, resource, time, traceback
HERE = Path(__file__).resolve().parent
OUT = HERE / 'output'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def serial(x):
    if isinstance(x, F): return str(x)
    if isinstance(x, dict): return {k: serial(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return list(map(serial, x))
    return x


def main():
    assert not OUT.exists() or not any(OUT.iterdir()), 'refuse any existing output'
    OUT.mkdir(exist_ok=True)
    start = time.process_time()
    r = dict(complete=False, pass_all=False, rows=[], author_model_calls=0, probability_calls=0)
    def save():
        r['cpu_seconds'] = time.process_time() - start
        r['maxrss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        tmp = OUT / 'results.tmp'
        tmp.write_text(json.dumps(serial(r), indent=2, allow_nan=False)+'\n')
        os.replace(tmp, OUT / 'results.json')
    def event(kind, **kw):
        assert time.process_time()-start < 15
        with (OUT/'progress.jsonl').open('a') as f: f.write(json.dumps(dict(kind=kind, **kw))+'\n')
    try:
        protocol = json.loads((HERE/'protocol.json').read_text())
        gate = json.loads((HERE/'gate.json').read_text())
        required = [HERE/'primitive.py', HERE/'run.py', HERE/'protocol.json', HERE/'PROTOCOL.md', HERE/'launch.sh', Path(protocol['source_review'])]
        assert gate['ready'] is True and all(str(p) in gate['pins'] for p in required)
        for p,h in {**protocol['pins'], **gate['pins']}.items(): assert sha(p)==h, p
        review=json.loads(Path(protocol['source_review']).read_text())
        assert review['complete'] is True and review['pass_all'] is True and review['execution_authorized'] is True
        assert all(str(p) in review['pins'] for p in required[:-1])
        for p,h in review['pins'].items(): assert sha(p)==h, p
        assert all(0<v<=20 for v in resource.getrlimit(resource.RLIMIT_CPU))
        assert all(0<v<=2*1024**3 for v in resource.getrlimit(resource.RLIMIT_AS))
        r['limits']={'cpu':resource.getrlimit(resource.RLIMIT_CPU),'as':resource.getrlimit(resource.RLIMIT_AS)}
        r['pins']=gate['pins']; event('started');save()
        from primitive import ownership, margin_status, mass_bounds
        def check(label, fn, args, expected, validator):
            row=dict(label=label,args=args,expected=expected,status='started');r['rows'].append(row);save();event('started_case',label=label)
            value=fn(*args);row['result']=value;save()
            validator(value, expected);row['status']='passed';save();event('completed_case',label=label)
        def box_check(v,e):
            assert v['inner_pairs']==e['inner']
            assert [a['pair'] for a in v['outer_pairs']]==e['outer']
            b,g=v['box'],v['gap']
            for a in v['outer_pairs']:
                i,j=a['pair'];x=a['witness'];assert all(l<=z<=u for z,(l,u) in zip(x,b))
                assert x[i]-x[j]>=g and all(x[j]-x[t]>=g for t in range(len(b)) if t not in(i,j))
            for i,j in v['inner_pairs']:
                for x in product(*b):
                    assert x[i]-x[j]>=g and all(x[j]-x[t]>=g for t in range(len(b)) if t not in(i,j))
        base=[[3,4],[1,2],[-1,0]]
        for n,p in enumerate(permutations(range(3))):
            pair=[p.index(0),p.index(1)]
            check('permutation'+str(n),ownership,[[base[j] for j in p],1],dict(inner=[pair],outer=[pair]),box_check)
        allpairs=[[i,j] for i in range(3) for j in range(3) if i!=j]
        special=[('overlap_zero',[[0,1]]*3,0,[],allpairs),('impossible',[[0,1]]*3,1,[],[]),
                 ('touch_only',[[0,1]]*3,'1/2',[],allpairs),('negative',[[-2,-1],[-4,-3],[-6,-5]],1,[[0,1]],[[0,1]]),
                 ('adjacent_zero',[[2,3],[1,2],[0,1]],0,[[0,1]],[[0,1],[0,2],[1,0]]),
                 ('four_inputs',[[3,4],[1,2],[-1,0],['-1/2','1/2']],'1/2',[[0,1]],[[0,1]])]
        for label,b,g,inn,outer in special:check(label,ownership,[b,g],dict(inner=inn,outer=outer),box_check)
        margins=[(0,[[0,0],[0,0],[0,0]],'correct'),(1,[[0,0],[0,0],[1,2]],'wrong'),
                 (2,[[0,0],[0,0],[0,0]],'wrong'),(1,[[1,2],[0,0],[0,0]],'correct'),
                 (1,[[-1,1],[0,0],[0,2]],'unknown'),(0,[[0,0],[-2,-1],[-1,1]],'wrong')]
        for n,(j,v,status) in enumerate(margins):check('margin'+str(n),margin_status,[j,v],status,lambda v,e: v['status']==e or (_ for _ in ()).throw(AssertionError(v)))
        masses=[(['3/4','4/5'],['1/4','3/10'],['1/10','1/8'],['13/40','9/20'],['5/16','7/8']),
                ([1,1],[1,1],[0,0],[0,0],[1,1]),([0,0],[0,0],[0,0],[0,0],None),
                (['1/2','3/4'],[0,1],[0,1],[0,'3/4'],[0,1])]
        def mass_check(v,e):
            assert v['unresolved']==list(map(F,e[0]))
            assert v['conditional_accuracy']==(None if e[1] is None else list(map(F,e[1])))
        for n,(p,c,w,u,a) in enumerate(masses):check('mass'+str(n),mass_bounds,[p,c,w],[u,a],mass_check)
        guards=[('negative_gap',ownership,[base,-1]),('zero_width',ownership,[[[0,0],[1,2],[3,4]],0]),
                ('two_inputs',ownership,[[[0,1],[2,3]],0]),('float_endpoint',ownership,[[[0.0,1],[2,3],[4,5]],0]),
                ('target_range',margin_status,[3,[[0,0]]*3]),('self_margin',margin_status,[0,[[1,1],[0,0]]]),
                ('negative_mass',mass_bounds,[[-1,0],[0,0],[0,0]]),('inconsistent_mass',mass_bounds,[[0,'1/2'],['2/5','1/2'],['2/5','1/2']]),
                ('reversed_mass',mass_bounds,[[1,0],[0,0],[0,0]])]
        for label,fn,args in guards:
            row=dict(label=label,args=args,status='started',expected='ValueError');r['rows'].append(row);save();event('started_case',label=label)
            try: value=fn(*args)
            except ValueError as e:row['error']=str(e);row['status']='passed'
            else:row['unexpected_result']=value;save();raise AssertionError('guard accepted '+label)
            save();event('completed_case',label=label)
        assert len(r['rows'])==31 and all(x['status']=='passed' for x in r['rows'])
        event('completed');r['complete']=r['pass_all']=True;save()
    except BaseException:
        r['complete']=r['pass_all']=False;r['failure']=traceback.format_exc();save();raise

if __name__=='__main__':main()
