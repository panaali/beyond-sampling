"""T227 deterministic saved-only contraction; no author imports or RNG."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from math import isfinite
import hashlib,json,time,resource,traceback,sys
HERE=Path(__file__).resolve().parent
MODELS=['z6s0','z6s1','z6s2','z6s3','z6s4','orig','s005','s010']
SCHEMES=['mech','pos','pos_flips','nopos','relpos','coarse'];LS=[10,30,100]
TOL=F(5,10**12)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def number(x):
    assert type(x) in (float,int) and isfinite(x),repr(x)
    return F.from_float(x) if type(x)is float else F(x)
def same(a,b,where):assert a==b,(where,a,b)
def sumq(xs):return sum(xs,F(0))
def atomic(p,d):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)

def label(p,T,scheme):
    kind,a,b,third=p['kind'],p['a'],p['b'],p['p1'];flips=tuple(tuple(x) for x in p['flips'])
    mx=third if kind=='23' else None
    age=T-1-(mx if mx is not None else max(a,b))
    xlast='max' if mx==T-1 else 'pair' if T-1 in (a,b) else 'third' if kind=='12' and third==T-1 else 'below'
    if scheme=='mech':return kind,a,b,third,p['ub'],p['wt'],flips
    if scheme=='pos':return kind,a,b,third
    if scheme=='pos_flips':return kind,a,b,third,flips
    if scheme=='nopos':return kind,p['ub'],p['wt'],flips
    if scheme=='relpos':return kind,abs(a-b),age,xlast,p['ub'],p['wt'],flips
    assert scheme=='coarse';return kind,p['ub'],p['wt'],min(p['nflips'],3)

def groups(points,T,scheme):
    out={}
    for p in points:out.setdefault(label(p,T,scheme),[]).append(number(p['g2']))
    return out

def ranked(groups):
    masses={k:sumq(v) for k,v in groups.items()}
    keys=sorted(masses,key=lambda k:masses[k],reverse=True)
    return keys,masses

def coverage(groups,chosen):
    masses={k:sumq(v) for k,v in groups.items()};den=sumq(masses.values());assert den>0
    return sumq(v for k,v in masses.items() if k in chosen)/den

def top_share(v):
    assert v;v=sorted(v,reverse=True);n=max(1,(len(v)+19)//20)
    return sumq(v[:n])/sumq(v)

def within(groups):
    selected=sorted((v for v in groups.values() if len(v)>=20),key=sumq,reverse=True)[:10]
    return dict(pooled=top_share([x for v in groups.values() for x in v]),labels=[top_share(v) for v in selected],npts=[len(v) for v in selected])

def main():
    out=HERE/'audit';assert not out.exists();out.mkdir();sys.set_int_max_str_digits(0)
    resource.setrlimit(resource.RLIMIT_CPU,(19,20));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3));start=time.process_time()
    r=dict(complete=False,models=[],cross_copy=[],prefix=[],comparisons=[],current=None,author_imports=0,model_calls=0,random_calls=0,LP_calls=0,mass_calls=0)
    def save():r.update(cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);atomic(out/'results.json',r)
    def emit(stage,**kw):
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,**kw),allow_nan=False)+'\n')
    def cap():assert time.process_time()-start<17,'17CPU internal stop to preserve failure before19/20 limit'
    def compare(value,saved,where):
        value=F(value);stored=number(saved);diff=abs(value-stored)
        r['comparisons'].append(dict(field=where,exact=str(value),saved=saved,absolute_difference=str(diff)))
        assert diff<=TOL,(where,str(value),saved,str(diff))
    try:
        protocol=read(HERE/'protocol.json');gate=read(HERE/'gate.json');assert gate['complete'];same(gate['protocol_sha256'],sha(HERE/'protocol.json'),'protocol')
        required={str(HERE/name) for name in ['check_saved.py','protocol.json','launcher.sh','source_manifest.json']}
        assert required <= set(gate['pins']), 'gate missing mandatory execution/source pins'
        same(set(gate['reviews']),{'source_provenance','checker_source'},'review prerequisites')
        for kind,item in gate['reviews'].items():
            assert item['commit'] and item['path'] in gate['pins']
            same(item['sha256'],gate['pins'][item['path']],kind+' pin')
            review=read(item['path']);assert review['complete'] is True and review['pass_all'] is True
            if kind=='source_provenance':
                same(item['commit'],'791b2d5f','provenance review commit')
                same(item['sha256'],'1b38d809c2e81a964b53d2d3e216efe1c2bfbf8fc1bf49dfc9a0229d5adb45ed','provenance review bytes')
        for path,h in {**protocol['pins'],**gate['pins']}.items():same(sha(path),h,path)
        manifest=read(HERE/'source_manifest.json')
        for entry in manifest['files']:same(sha(HERE.parents[1]/entry['local_path']),entry['sha256'],entry['local_path'])
        same(protocol['models'],MODELS,'models');same(protocol['schemes'],SCHEMES,'schemes');same(protocol['L'],LS,'L');same(F(protocol['absolute_tolerance']),TOL,'tolerance')
        r.update(protocol_sha256=sha(HERE/'protocol.json'),gate_sha256=sha(HERE/'gate.json'));emit('started');save()
        frozen=HERE/'frozen';summary=read(frozen/'summary.json');blind=read(frozen/'results/blind_split.json');tails={};groups_cache={}
        same(set(summary),set(MODELS+['cross_copy_from_orig']),'summary models');same(set(blind),set(MODELS),'blind models')
        for name in MODELS:
            cap();r['current']=dict(model=name,stage='raw');save();d=read(frozen/'results'/f'{name}.json');s=summary[name];T=6 if name.startswith('z6s') else 10
            same((d['model'],d['T'],d['n']),(name,T,16384),'model metadata');same((s['T'],s['n']),(T,16384),'summary metadata');same(set(d['rules']),{'700','701'},'rules')
            model=dict(name=name,T=T,rules=[],schemes={},primary_decision=None);r['models'].append(model);tails[name]={}
            for rule_idx,key in enumerate(['700','701']):
                raw=d['rules'][key];points=raw['tail'];same(len(points),1639,'tail size');weights=[number(p['g2']) for p in points];assert all(v>=0 for v in weights)
                same(weights,sorted(weights,reverse=True),'tail ordering');total=sumq(weights);compare(total,raw['tail_g2'],f'{name}/{key}/tail_g2')
                denominator=number(raw['sum_g2']);assert denominator>0 and total<=denominator+TOL
                compare(total/denominator,raw['top5pct_share'],f'{name}/{key}/top5');compare(sumq(weights[:327])/denominator,raw['top1pct_share'],f'{name}/{key}/top1')
                compare(number(raw['err'])*2,s['err'][rule_idx],f'{name}/{key}/summary_error');same(s['top5'][rule_idx],raw['top5pct_share'],'summary top5');same(s['top1'][rule_idx],raw['top1pct_share'],'summary top1')
                assert type(raw['nzero'])is int and 0<=raw['nzero']<=32768
                retained=[];missing=[];duplicates=0;unordered=0;checks=[]
                for i,p in enumerate(points):
                    r['current']=dict(model=name,rule=key,point=i,stage='schema');checks.append(number(p['Fcheck']));assert checks[-1]>=0
                    if not p.get('kind'):missing.append(number(p['g2']));continue
                    assert p['kind'] in ['12','23'];a,b,c=p['a'],p['b'],p['p1'];assert all(type(v)is int and 0<=v<T for v in [a,b,c]) and a<b and c not in [a,b]
                    assert type(p['npos'])is int and 0<=p['npos']<=T and type(p['ncomp'])is int and p['ncomp']>=1
                    assert type(p['nflips'])is int and p['nflips']>=0 and len(p['flips'])==min(3,p['nflips'])
                    fs=[tuple(q) for q in p['flips']];assert all(len(q)==2 and all(type(v)is int for v in q) and 0<=q[0]<T and 0<=q[1]<16 for q in fs)
                    duplicates+=len(fs)!=len(set(fs));unordered+=fs!=sorted(fs)
                    u=number(p['u']);bucket=0 if u<=0 else 1 if u<=F(1,4) else 2 if u<=F(1,2) else 3;same(p['ub'],bucket,'u bucket');same(p['npos']<=1,u<=0,'face sign')
                    assert p['side'] in ['+','-'] and p['wt'] in (['tie','max','other'] if p['kind']=='23' else ['tie','third','other']) and number(p['width'])>=0
                    retained.append(p)
                tails[name][key]=retained
                model['rules'].append(dict(rule=key,total_points=1639,retained=len(retained),missing_kind=len(missing),missing_g2=str(sumq(missing)),max_saved_Fcheck=str(max(checks)),retained_duplicate_flip_lists=duplicates,retained_non_sorted_flip_lists=unordered,total_tail_g2=str(total)))
            same(set(s['schemes']),set(SCHEMES),'exact summary scheme coverage')
            same(s['ntail'],len(tails[name]['700']),'ntail');compare(blind[name]['err'],s['err'][0],f'{name}/blind-error-summary');compare(blind[name]['top5_all'],s['top5'][0],f'{name}/blind-top5-summary')
            for scheme in SCHEMES:
                cap();r['current']=dict(model=name,scheme=scheme,stage='deterministic contraction')
                A=groups(tails[name]['700'],T,scheme);B=groups(tails[name]['701'],T,scheme);AB=groups(tails[name]['700']+tails[name]['701'],T,scheme)
                groups_cache[name,scheme]=(A,B);ka,ma=ranked(A);kb,mb=ranked(B);kab,mab=ranked(AB);prior=s['schemes'][scheme]
                same(prior['nlabels_A'],len(A),'nlabels_A');same(prior['nlabels_AB'],len(AB),'nlabels_AB')
                res=dict(nlabels_A=len(A),nlabels_AB=len(AB),selected_A100=[dict(label=k,mass=str(ma[k])) for k in ka[:100]],selected_B100=[dict(label=k,mass=str(mb[k])) for k in kb[:100]],metrics={},selection_gaps={})
                model['schemes'][scheme]=res
                for L in LS:
                    chosenA=set(ka[:L]);chosenB=set(kb[:L]);chosenAB=set(kab[:L])
                    metrics={f'in_A@{L}':coverage(A,chosenA),f'oos_AB@{L}':coverage(B,chosenA),f'oos_BA@{L}':coverage(A,chosenB),f'in_AB@{L}':coverage(AB,chosenAB)}
                    for field,value in metrics.items():res['metrics'][field]=str(value);compare(value,prior[field],f'{name}/{scheme}/{field}')
                    res['selection_gaps'][str(L)]=dict(A=None if len(ka)<=L else str(ma[ka[L-1]]-ma[ka[L]]),B=None if len(kb)<=L else str(mb[kb[L-1]]-mb[kb[L]]))
                w=within(A);same(w['npts'],prior['within']['npts'],'within sizes');compare(w['pooled'],prior['within']['pooled'],f'{name}/{scheme}/within/pooled');same(len(w['labels']),len(prior['within']['labels']),'within labels')
                for j,v in enumerate(w['labels']):compare(v,prior['within']['labels'][j],f'{name}/{scheme}/within/{j}')
                res['within']=dict(pooled=str(w['pooled']),labels=list(map(str,w['labels'])),npts=w['npts']);save();emit('scheme_complete',model=name,scheme=scheme)
            singleton=sorted((number(p['g2']) for p in tails[name]['700']),reverse=True)
            for L in LS:compare(sumq(singleton[:L])/sumq(singleton),s['null_singleton'][str(L)],f'{name}/singleton/{L}')
            primary=F(model['schemes']['mech']['metrics']['oos_AB@100']);model['primary_below_70pct']=primary<F(7,10)
            model['unreconstructed_fields']=['in_half@L','null_random','composition rounded display','all-face blind/CV statistics except two shared-summary consistency checks','original labels/walk Fcheck truth']
            save();emit('model_complete',model=name)
        for scheme in ['mech','pos','relpos','nopos']:
            A,_=groups_cache['orig',scheme];keys,_=ranked(A)
            for name in ['orig','s005','s010']:
                _,B=groups_cache[name,scheme]
                for L in LS:
                    value=coverage(B,set(keys[:L]));key=f'{name}@{L}';r['cross_copy'].append(dict(scheme=scheme,key=key,exact=str(value)));compare(value,summary['cross_copy_from_orig'][scheme][key],f'cross/{scheme}/{key}')
        for filename,full,kmax in [('prefix_blind.json',False,4),('prefix_blind_full.json',True,3)]:
            d=read(frozen/'results'/filename);same(d['full'],full,'prefix full');same(d['kmax'],kmax,'prefix kmax');same([x['k'] for x in d['steps']],list(range(1,kmax+1)),'prefix coverage')
            for j,entry in enumerate(d['steps']):
                assert entry['complete'] is True and all(type(entry[k])is int and entry[k]>=0 for k in ['cells','regions','lps']) and entry['regions']<=entry['cells'] and number(entry['sec'])>=0
                if j:
                    for kind in ['cells','regions']:
                        compare(F(entry[kind],d['steps'][j-1][kind]),entry['growth_'+kind],f'{filename}/{entry["k"]}/growth_{kind}')
            r['prefix'].append(dict(file=filename,steps=d['steps'],scope='Saved summary arithmetic only; no exact feasibility/history/cap/LP verification.'))
        r['primary_fixed_rule_kill']=any(next(m for m in r['models'] if m['name']==n)['primary_below_70pct'] for n in ['orig','s005','s010'])
        same(len(r['models']),8,'models complete');same(len(r['cross_copy']),36,'cross copy complete');cap();r['current']=None;r['complete']=True;save();emit('completed',results_sha256=sha(out/'results.json'))
        print(json.dumps(dict(complete=True,models=8,comparisons=len(r['comparisons']),primary_fixed_rule_kill=r['primary_fixed_rule_kill'],cpu_seconds=r['cpu_seconds'])))
    except BaseException as e:r.update(failure=repr(e),traceback=traceback.format_exc());save();emit('failed',error=repr(e));raise
if __name__=='__main__':main()
