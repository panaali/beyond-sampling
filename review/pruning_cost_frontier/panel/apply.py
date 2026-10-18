"""One fixed retained binary-value oracle; no NumPy, model or mass calls."""
import ast, gzip, hashlib, json, math, resource, struct, time, traceback, zipfile
from fractions import Fraction as F
from pathlib import Path
P=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def enc(x):
    if isinstance(x,F):return [str(x.numerator),str(x.denominator)]
    if isinstance(x,float) and not math.isfinite(x):return {'nonfinite_float':x.hex()}
    if isinstance(x,dict):return {k:enc(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [enc(v) for v in x]
    return x

def arrays(path):
    expected={'err','mass_ops','simplices','kind','leaf_mass','rays'}
    with zipfile.ZipFile(path) as z:
        assert len(z.infolist())==6 and set(z.namelist())=={k+'.npy' for k in expected}
        out={};headers={}
        for k in sorted(expected):
            info=z.getinfo(k+'.npy');assert info.file_size<=100000*8+65536
            raw=z.read(info);assert raw[:6]==b'\x93NUMPY'
            version=tuple(raw[6:8]);assert version in [(1,0),(2,0)]
            width=2 if version==(1,0) else 4
            size=int.from_bytes(raw[8:8+width],'little');assert 0<size<=65536
            start=8+width+size;header=ast.literal_eval(raw[8+width:start].decode('latin1'))
            assert set(header)=={'descr','fortran_order','shape'}
            assert header['descr']=='<f8' and header['fortran_order'] is False
            shape=header['shape'];assert isinstance(shape,tuple) and len(shape)==1 and type(shape[0]) is int and 0<shape[0]<=100000
            assert len(raw)==start+8*shape[0]
            out[k]=[x[0] for x in struct.iter_unpack('<d',raw[start:])]
            headers[k]={'version':version,'header':header,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    assert len({len(v) for v in out.values()})==1
    return out,headers

def exact_int(v):
    assert not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v) and v>=0 and int(v)==v and v<=2**53
    return int(v)

def main():
    out=P/'target_results.json';progress=P/'target_progress.jsonl';rawpath=P/'target_frontier.json.gz'
    assert not any(x.exists() for x in [out,progress,rawpath,P/'target_frontier.tmp'])
    start=time.process_time();r={'complete':False,'pass_all':False,'frontier_attempts':0,'frontier_returns':0,'target_mass_calls':0,'model_calls':0}
    def save():
        r['cpu_seconds']=time.process_time()-start;t=P/'target_results.tmp'
        t.write_text(json.dumps(enc(r),indent=2,allow_nan=False)+'\n');t.replace(out)
    def event(name):
        with progress.open('a') as f:f.write(json.dumps({'event':name,'cpu_seconds':time.process_time()-start})+'\n')
        if time.process_time()-start>=17:raise RuntimeError('internal CPU cap')
    try:
        p=load(P/'target_protocol.json');g=load(P/'target_gate.json')
        required=[str(P/n) for n in ['apply.py','primitive.py','target_protocol.json','TARGET_PROTOCOL.md','retained/manifest.json']]
        assert g['ready'] is True and all(v in g['pins'] for v in required)
        for key in ['source_publication','review_commit','review_publication','parity_review_commit','parity_review_publication']:
            s=g[key];assert isinstance(s,str) and 8<=len(s)<=40 and all(c in '0123456789abcdef' for c in s)
        assert g['source_review'] in g['pins'] and g['parity_review'] in g['pins']
        for pins in [p['pins'],g['pins']]:
            for f,h in pins.items():assert sha(f)==h,f
        rv=load(g['source_review']);assert all(rv[k] is True for k in ['complete','pass_all','execution_allowed'])
        assert all(f in rv['pins'] for f in required)
        for f,h in rv['pins'].items():assert sha(f)==h,f
        parity=load(g['parity_review']);assert parity['complete'] is True and parity['pass_all'] is True
        assert parity['pins'][str(P/'results.json')]==p['pins'][str(P/'results.json')]
        prior=load(P/'results.json');assert prior['complete'] is True and prior['pass_all'] is True
        r['limits']={'cpu':resource.getrlimit(resource.RLIMIT_CPU),'address_space':resource.getrlimit(resource.RLIMIT_AS)};save()
        assert r['limits']=={'cpu':(20,20),'address_space':(2**31,2**31)}
        event('begin')
        manifest=load(P/'retained/manifest.json')
        for item in manifest['records']:
            b=Path(item['staged_path']).read_bytes()
            assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
            assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==item['blob']
        name='H16_T4_zoo0_chamber'
        a,headers=arrays(P/'retained'/f'{name}.npz')
        meta=load(P/'retained'/f'{name}.json');historical=load(P/'retained/anatomy_summary.json')[name]
        n=len(a['err'])
        r['decoded']={'rows':n,'headers':headers,'json':meta,'historical_summary':historical};save();event('decoded')
        counts={k:{'nonfinite':sum(not math.isfinite(v) for v in vals),'negative':sum(v<0 for v in vals)} for k,vals in a.items()}
        r['input_counts']=counts;save()
        assert all(z['nonfinite']==0 for z in counts.values())
        assert counts['err']['negative']==0 and counts['mass_ops']['negative']==0
        c=[exact_int(v) for v in a['mass_ops']]
        for k in ['simplices','rays']:assert all(exact_int(v)==v for v in a[k])
        assert all(v in (0.,1.,2.) for v in a['kind'])
        O=exact_int(meta['ops']);M=sum(c);assert M==exact_int(meta['mass_ops']) and n==exact_int(meta['leaves'])
        assert O-M>0 and exact_int(meta['geometry_ops'])==O-M
        for k in ['ops','mass_ops','geometry_ops','leaves','acc','err_total']:assert historical[k]==meta[k]
        p0=meta['acc'];assert isinstance(p0,float) and math.isfinite(p0) and 0<p0<1
        prob=F.from_float(p0);e=[F.from_float(v) for v in a['err']];esum=sum(e,F())
        cs=4*(2*16*16+2*16)+2*16*4+4*16+3*4-2;assert cs==2378
        Q=prob*(1-prob)*cs/2
        r['validation']={'positive_fixed_cost':O-M,'mass_cost_sum':M,'zero_cost_rows':sum(v==0 for v in c),'zero_cost_positive_error_rows':sum(ci==0 and ei>0 for ci,ei in zip(c,e)),'exact_error_sum':esum,'error_sum_minus_stored_total':esum-F.from_float(meta['err_total']),'accuracy_minus_one_minus_error_sum':prob-(1-esum),'sampler_cost':cs,'threshold_Q':Q}
        save();event('validated')
        from primitive import frontier
        r['frontier_attempts']=1;save();event('frontier_attempt')
        ans=frontier(e,c,O,Q,80);r['frontier_returns']=1;save();event('frontier_return')
        # Preserve the complete returned oracle before any post-return comparisons.
        temp=P/'target_frontier.tmp'
        with gzip.open(temp,'wt',encoding='utf8',compresslevel=1) as f:json.dump(enc(ans),f,separators=(',',':'),allow_nan=False)
        temp.replace(rawpath);r['raw_frontier_sha256']=sha(rawpath);save();event('raw_saved')
        methods={}
        for key,z in ans['methods'].items():
            best=z['best'];s=best['saved'];bias=best['error'];remaining=2*(O-s)
            assert bias*bias*(O-s)<=Q
            methods[key]={'selected_count':len(best['indices']),'saved_ops':s,'remaining_nominal_flops':remaining,'remaining_fraction':(O-s)/O,'bias':bias,'bias_squared_over_nominal_sampling_variance':bias*bias*(O-s)/Q,'bias_float_diagnostic':float(bias),'nominal_sampling_sd_float_diagnostic':math.sqrt(float(Q/(O-s)))}
        feasible=max(z['saved_ops'] for z in methods.values());upper=ans['fractional_saved_upper']
        assert feasible<=upper<=M
        last=ans['breakpoints'][ans['last_feasible_breakpoint']]['saved'];ratio=methods['ratio_self']['saved_ops']
        assert ratio>=last
        largest=max(c,default=0);assert upper-ratio<=largest
        r['summary']={'methods':methods,'fractional_saved_lower':ans['fractional_saved_lower'],'fractional_saved_upper':upper,'best_integer_feasible_saved':feasible,'integer_saved_gap_upper':upper-feasible,'integer_remaining_flop_bracket':[2*(O-upper),2*(O-feasible)],'original_nominal_flops':2*O,'fixed_nominal_flops':2*(O-M),'largest_leaf_mass_ops':largest,'historical_skip_at_1se':historical['skip_at_1se'],'qualification':'Exact retained binary-value oracle only; historical NumPy/tie/summation report is separate; no true-mass, deployment or full-native-cost certificate.'}
        save();event('complete');r.update(complete=True,pass_all=True);save()
    except BaseException:
        r.update(complete=False,pass_all=False,failure=traceback.format_exc());save();raise

if __name__=='__main__':main()
