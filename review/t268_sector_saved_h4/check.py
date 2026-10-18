"""Independent exact elimination and saved interval/sector audit. No author calls."""
from pathlib import Path
from fractions import Fraction as F
import gzip,hashlib,json,math,resource,sys,time,traceback
P=Path(__file__).resolve().parent;OUT=P/'output'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unpack(v):
    if type(v)is dict:
        if set(v)=={'q'}:return F(*map(int,v['q']))
        return {k:unpack(x) for k,x in v.items()}
    if type(v)is list:return [unpack(x) for x in v]
    return v

def determinant(rows):
    a=[[F(x) for x in r] for r in rows];answer=F(1)
    for k in range(4):
        pivot=next((j for j in range(k,4) if a[j][k]),None)
        if pivot is None:return F(0)
        if pivot!=k:a[k],a[pivot]=a[pivot],a[k];answer=-answer
        z=a[k][k];answer*=z
        for j in range(k+1,4):
            t=a[j][k]/z
            for col in range(k+1,4):a[j][col]-=t*a[k][col]
    return answer

def sector(rows):
    if type(rows)is not list or len(rows)!=4 or any(type(r)is not list or len(r)!=4 for r in rows):return 'four by four rays'
    if any(type(x)is not int for r in rows for x in r):return 'integer rays'
    key=0
    for col in range(4):
        signs={1 if r[col]>0 else -1 for r in rows if r[col]}
        if len(signs)==2:return 'mixed coordinate signs'
        if not signs:return 'zero coordinate span'
        if 1 in signs:key+=2**col
    d=determinant(rows)
    if not d:return 'singular simplex'
    scale=math.prod(sum(abs(x) for x in r) for r in rows)
    return [key,abs(d)/scale]

def law(a,b):
    p=F(1,16);mid=p/2
    if not 0<=a<=b<=p:return 'physical probability interval'
    near=F(0) if a<=mid<=b else min(abs(a-mid),abs(b-mid));far=max(abs(a-mid),abs(b-mid))
    return {'correct':[a,b],'error':[p-b,p-a],'conditional_correct':[16*a,16*b],'best_binary_loss':[mid-far,mid-near],'strictly_nonconstant':a>0 and b<p}

def main():
    assert not OUT.exists();OUT.mkdir();start=time.process_time();r={'complete':False,'pass_all':False,'rows':0,'controls':0,'source_calls':0}
    def save():
        r.update(cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        f=OUT/'results.tmp';f.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');f.replace(OUT/'results.json')
    try:
        save();s=read(P/'scope.json');g=read(P/'gate.json');assert g['ready'] is True
        for k in ['source_publication','review_publication','execution_publication']:assert isinstance(g[k],str) and 8<=len(g[k])<=40 and all(x in '0123456789abcdef' for x in g[k])
        required={str(P/f) for f in ['check.py','scope.json','README.md']}|set(s['pins'])|{s['source_review']};assert required<=set(g['pins'])
        for pins in [s['pins'],g['pins']]:
            for f,h in pins.items():assert sha(f)==h,f
        rv=read(s['source_review']);assert all(rv[k] for k in ['complete','pass_all','execution_allowed'])
        assert {str(P/f) for f in ['check.py','scope.json','README.md']}<=set(rv['pins'])
        for f,h in rv['pins'].items():assert sha(f)==h,f
        assert resource.getrlimit(resource.RLIMIT_CPU)==(19,20) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,2**31)
        assert str(Path(sys.executable).resolve())==s['python_resolved'];owner=Path(s['owner']);p=read(owner/'protocol.json');raw=unpack(read(owner/'output/results.json'))
        assert raw['complete'] and raw['pass_all'];assert set(raw['pins'])<=set(g['pins'])
        for f,h in raw['pins'].items():assert g['pins'][f]==h
        r['pins']=g['pins'];save()
        # Literal fixture coverage, independently reconstructed from frozen specification.
        diag=lambda signs:[[signs[i] if i==j else 0 for j in range(4)] for i in range(4)]
        orth=[('positive',diag([1]*4),[15,F(1)]),('negative',diag([-1]*4),[0,F(1)]),('sole_first',diag([1,-1,-1,-1]),[1,F(1)]),('boundary',[[1,0,0,0],[1,-1,0,0],[0,0,-1,0],[0,0,0,-1]],[1,F(1,2)])]
        guards=[('mixed',[[1,0,0,0],[-1,1,0,0],[0,0,1,0],[0,0,0,1]],'mixed coordinate signs'),('zero_axis',[[1,0,0,0],[0,1,0,0],[0,0,1,0],[1,1,1,0]],'zero coordinate span'),('singular',[[1,1,1,1]]*4,'singular simplex'),('shape',[[1,0,0,0]]*3,'four by four rays')]
        p16=F(1,16);laws=[('zero',F(0),F(0)),('all',p16,p16),('low',p16/4,p16/3),('high',2*p16/3,3*p16/4),('cross_half',p16/4,3*p16/4),('half',p16/2,p16/2)]
        bad=[('inverted',p16,F(0)),('negative',-p16,F(0)),('above',F(0),2*p16)]
        assert len(raw['controls'])==17
        for c,(name,rows,want) in zip(raw['controls'][:4],orth,strict=True):assert c['name']==name and c['kind']=='orthant' and c['rays']==rows and c['expected']==c['value']==want==sector(rows) and c['passed']
        for c,(name,rows,msg) in zip(raw['controls'][4:8],guards,strict=True):assert c['name']==name and c['kind']=='orthant_guard' and c['rays']==rows and c['rejection']==sector(rows)==msg and c['passed']
        for c,(name,a,b) in zip(raw['controls'][8:14],laws,strict=True):assert c['name']==name and c['kind']=='law' and c['interval']==[a,b] and c['value']==law(a,b) and c['expected_loss']==law(a,b)['best_binary_loss'] and c['passed']
        for c,(name,a,b) in zip(raw['controls'][14:],bad,strict=True):assert c['name']==name and c['kind']=='law_guard' and c['interval']==[a,b] and c['rejection']==law(a,b) and c['passed']
        r['controls']=17;save()
        geometry=json.loads(gzip.decompress(Path(p['geometry']).read_bytes()));ref=read(p['reference']);simplices=geometry['correct_simplices'];bounds=ref['per_simplex_intervals'];assert len(simplices)==len(bounds)==5058
        assert ref['complete'] and ref['H']==ref['T']==4 and ref['simplices']==5058 and ref['exact_artifact_sha256']==sha(p['geometry'])
        assert raw['checkpoint_sha256']==geometry['checkpoint_sha256'] and raw['inherited_reference_cost']==ref['cold_cost'] and raw['reference_classification']==ref['classification'] and raw['inherited_reference_interval']==ref['accuracy']['probability_interval']
        counts=[0]*16;low=[F(0)]*16;high=[F(0)]*16;rawlo=rawhi=F(0)
        assert len(raw['chunks'])==20
        for j,chunk in enumerate(raw['chunks']):
            assert time.process_time()-start<17
            path=owner/'output'/f'chunk_{j:02d}.jsonl';n=min(256,5058-j*256)
            assert chunk=={'index':j,'first':j*256,'rows':n,'path':path.name,'sha256':sha(path)}
            records=[unpack(json.loads(line)) for line in path.read_text().splitlines()];assert len(records)==n
            for k,row in enumerate(records):
                i=256*j+k;item=simplices[i];rays=[[int(x) for x in z] for z in item['integer_rays']]
                assert [[str(x) for x in z] for z in rays]==item['integer_rays'];sid,det=sector(rays);assert det==F(*map(int,item['normalized_absolute_determinant']))
                v=bounds[i];a,b=v['quadrature'];e=v['truncation_upper'];assert all(type(x)is float and math.isfinite(x) for x in [a,b,e]) and 0<=a<=b and e>=0
                a,b=F(a)-F(e),F(b)+F(e);rawlo+=a;rawhi+=b;cl,cu=max(F(0),a),min(p16,b);assert cl<=cu
                assert row=={'index':i,'sector':sid,'normalized_determinant':det,'raw_interval':[a,b],'physical_interval':[cl,cu]}
                counts[sid]+=1;low[sid]+=cl;high[sid]+=cu;r['rows']+=1
            save()
        assert raw['raw_global_interval']==[rawlo,rawhi];reflo,refhi=map(F,ref['accuracy']['probability_interval']);assert raw['reference_global_exact']==[reflo,refhi] and reflo<=rawlo<=rawhi<=refhi
        expected=[]
        for j in range(16):
            a,b=max(F(0),low[j]),min(p16,high[j]);z=law(a,b);assert isinstance(z,dict)
            expected.append({'sector':j,'positive_positions':[i for i in range(4) if j//2**i%2],'simplex_count':counts[j],'unclipped_sum':[low[j],high[j]],'sector_mass':p16,**z})
        assert raw['sectors']==expected
        blind=[expected[j] for j in [0,1,2,4,8]];c=[sum(z['correct'][i] for z in blind) for i in range(2)];loss=[sum(z['best_binary_loss'][i] for z in blind) for i in range(2)];mass=F(5,16)
        assert raw['blind']=={'sector_ids':[0,1,2,4,8],'mass':mass,'correct':c,'error':[mass-c[1],mass-c[0]],'conditional_accuracy':[x/mass for x in c],'best_binary_loss':loss,'nonconstant_sector_ids':[z['sector'] for z in blind if z['strictly_nonconstant']]}
        events=[json.loads(line) for line in (owner/'output/progress.jsonl').read_text().splitlines()];wanted=[{'event':'started'}]
        for i in range(17):wanted += [{'event':'control_started','index':i},{'event':'control_completed','index':i}]
        for j in range(20):wanted += [{'event':'chunk_started','index':j,'first':j*256},{'event':'chunk_completed','index':j}]
        wanted += [{'event':'completed'}];assert events==wanted and raw['events']==len(events)==76
        for key,value in [('rows',5058),('orthant_attempts',5066),('law_attempts',25),('model_calls',0),('geometry_generated',0),('mass_calls',0),('rng_calls',0)]:assert raw[key]==value
        assert 0<=raw['contraction_cpu']<=raw['cpu_seconds']<17 and raw['maxrss_kib']<2**21
        r.update(sectors=16,blind_sectors=5,events=76,producer_cpu_seconds=raw['cpu_seconds'],blind_summary={k:[[str(x.numerator),str(x.denominator)] for x in raw['blind'][k]] for k in ['correct','error','conditional_accuracy','best_binary_loss']},strictly_nonconstant=raw['blind']['nonconstant_sector_ids'],qualification='Saved reference contraction only; no independent mass/geometry/partition/model replay. Variable correctness with varying truth labels does not prove variable predicted class or exclude an accuracy functional.')
        assert r['rows']==5058;r['complete']=r['pass_all']=True;save()
    except BaseException:r.update(complete=False,pass_all=False,failure=traceback.format_exc());save();raise
if __name__=='__main__':main()
