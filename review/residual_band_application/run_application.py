"""Fixed T278 H4T4 numerical discriminator; no adaptive accuracy choices."""
from pathlib import Path
from fractions import Fraction as F
import gzip,hashlib,importlib.util,itertools,json,math,resource,sys,time,traceback
P=Path(__file__).resolve().parent;OUT=P/'application_output'
FILES=['run_application.py','application_math.py','application_protocol.json','APPLICATION.md','application_launch.sh']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def encode(x):
    if isinstance(x,F):return {'q':[str(x.numerator),str(x.denominator)]}
    if isinstance(x,dict):return {str(k):encode(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [encode(v) for v in x]
    return x

def main():
    assert not OUT.exists();OUT.mkdir();started=time.process_time()
    result={'complete':False,'pass_all':False,'points':0,'events':0,'source_face_calls':0,'model_partition_calls':0,'rng_calls':0,'reference_recomputed':False,'summary':[],'cpu_phases':{},'raw_bytes':0}
    def save():
        result['cpu_seconds']=time.process_time()-started;result['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        tmp=OUT/'results.tmp';tmp.write_text(json.dumps(encode(result),indent=2,allow_nan=False)+'\n');tmp.replace(OUT/'results.json')
    def snapshot(name,value):
        tmp=OUT/(name+'.tmp');tmp.write_text(json.dumps(encode(value),separators=(',',':'),allow_nan=False)+'\n');tmp.replace(OUT/(name+'.json'))
    def cap():
        if time.process_time()-started>=90:raise RuntimeError('internal CPU cap')
        if result['raw_bytes']>128*2**20:raise RuntimeError('raw128MiB cap')
    def event(kind,**kw):
        cap()
        with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps({'event':kind,'cpu':time.process_time()-started,**kw})+'\n')
        result['events']+=1;save()
    def timed(name,fun,*args,**kw):
        cap();t=time.process_time();v=fun(*args,**kw);dt=time.process_time()-t
        result['cpu_phases'][name]=result['cpu_phases'].get(name,0.)+dt;return v,dt
    try:
        save();p=read(P/'application_protocol.json');g=read(P/'application_gate.json');assert g['ready'] is True
        own=[str(P/n) for n in FILES];required=own+list(p['pins'])+[p['source_review'],p['controls_review']]
        assert all(f in g['pins'] for f in required)
        for name in ['source_publication','review_publication','execution_publication']:
            x=g[name];assert isinstance(x,str) and 8<=len(x)<=40 and all(c in '0123456789abcdef' for c in x)
        for pins in (p['pins'],g['pins']):
            for f,h in pins.items():assert sha(f)==h,f
        review=read(p['source_review']);assert review['complete'] and review['pass_all'] and review['execution_authorized']
        assert all(f in review['pins'] for f in own)
        for f,h in review['pins'].items():assert sha(f)==h,f
        oldreview=read(p['controls_review']);assert oldreview['complete'] and oldreview['pass_all']
        assert oldreview['rows']==18 and oldreview['events']==38
        for f,h in oldreview['input_pins'].items():assert sha(f)==h,f
        controls=read(P/'controls_output/results.json');assert controls['complete'] and controls['pass_all'] and len(controls['rows'])==18 and controls['events']==38
        assert resource.getrlimit(resource.RLIMIT_CPU)==(105,110) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,2**31)
        assert str(Path(sys.executable).resolve())==p['python_resolved']
        import numpy as np
        import scipy
        assert np.__version__==p['numpy_version'] and scipy.__version__==p['scipy_version']
        from intervals import Ledger,partition,errors,signed_difference
        from gaussian import mass
        from application_math import evaluate
        result['pins']=g['pins'];result['runtime']={'python':sys.version,'numpy':np.__version__};event('started')
        # Formula parity uses only completed synthetic bands, before target loading.
        parity=[];result['synthetic_formula_parity']=parity
        for row in controls['rows']:
            if row['kind']!='gaussian':continue
            a,b,sign=row['band'];a=F(*map(int,a['q']));b=F(*map(int,b['q']));checks={};parity.append(checks);save()
            for mode in ['first','second','direct']:
                value,_=evaluate([(a,b,sign)],mode);checks[mode]=value.hex();save();assert value.hex()==row['value'][mode].hex()
        assert len(parity)==4;result['synthetic_formula_parity']=parity;event('formula_parity_completed')
        t=time.process_time();artifact=json.loads(gzip.decompress(Path(p['model_archive']).read_bytes()))
        def dec(pair):return F(*map(int,pair))
        model=([dec(v) for v in artifact['model'][0]],[[dec(v) for v in r] for r in artifact['model'][1]],[[dec(v) for v in r] for r in artifact['model'][2]])
        assert len(model[0])==4 and len(model[1])==4 and len(model[2])==4 and all(len(r)==4 for r in model[1]+model[2])
        result['model']={'weights':model,'checkpoint_sha256':artifact['checkpoint_sha256']};del artifact
        reference=read(p['reference']);assert reference['complete'] and reference['H']==4 and reference['T']==4
        assert reference['exact_artifact_sha256']==p['pins'][p['model_archive']]
        reflo,refhi=reference['accuracy']['probability_interval'];assert 0<reflo<refhi<1
        assert [reflo.hex(),refhi.hex()]==reference['accuracy']['probability_interval_hex']
        result['reference_interval']=[reflo,refhi];result['reference_scope']='Reused exact-geometry/conditional numerical enclosure, not recomputed.'
        model_cpu=time.process_time()-t;result['model_and_reference_load_cpu']=model_cpu;save()
        t=time.process_time();spec=importlib.util.spec_from_file_location('frozen_t278_face',p['face_source']);face=importlib.util.module_from_spec(spec);spec.loader.exec_module(face)
        tables,z23,z12=face.tables(4);configs={k:face.configs(4,k) for k in ['23','12']};assert [len(configs[k]) for k in ['23','12']]==[12,6]
        table_cpu=time.process_time()-t;result['face_setup']={'Z':[float(z23),float(z12)],'cpu':table_cpu,'grid_points':160001,'scope':'Frozen[-8,8]tabulated/interpolated CDF pushforward; not exact ideal face law.'}
        rules={};rule_cpu={}
        for order in [3,5]:
            t=time.process_time();n,w=np.polynomial.legendre.leggauss(order);n=(n+1)/2;w=w/2
            assert np.all(w>0) and abs(float(w.sum())-1)<1e-14
            rules[order]=(n,w);rule_cpu[order]=time.process_time()-t
        result['rules']={str(o):{'nodes':[float(x).hex() for x in rules[o][0]],'weights':[float(x).hex() for x in rules[o][1]],'setup_cpu':rule_cpu[o]} for o in rules};save()
        catalogue=[];lookup={};members={3:[],5:[]}
        def register(kind,ci,uvw):
            key=(kind,ci,tuple(float(x).hex() for x in uvw))
            if key not in lookup:lookup[key]=len(catalogue);catalogue.append({'kind':kind,'config':ci,'unit_hex':list(key[2])})
            return lookup[key]
        for kind in ['23','12']:
            for ci in range(len(configs[kind])):register(kind,ci,[.5]*3)
        for order,(nodes,weights) in rules.items():
            for kind,Z in [('23',z23),('12',z12)]:
                for ci in range(len(configs[kind])):
                    for ids in itertools.product(range(order),repeat=3):
                        uvw=[float(nodes[j]) for j in ids];idx=register(kind,ci,uvw);weight=float(Z)/len(configs[kind])*math.prod(float(weights[j]) for j in ids)
                        assert weight>0;members[order].append((idx,weight))
        assert len(catalogue)==2718 and len(members[3])==486 and len(members[5])==2250
        full_ids={j*2717//63 for j in range(64)};assert len(full_ids)==64
        (OUT/'catalogue.json').write_text(json.dumps({'catalogue':catalogue,'members':members,'full_cell_record_ids':sorted(full_ids)},indent=2)+'\n')
        result['catalogue_sha256']=sha(OUT/'catalogue.json');result['full_cell_record_ids']=sorted(full_ids);save()
        perpoint=[];max_window_difference=0.
        for idx,item in enumerate(catalogue):
            result['active_point']={'index':idx,**item,'phase':'face'};event('point_started',index=idx);kind=item['kind'];ci=item['config'];a,b,top=map(int,configs[kind][ci]);U=np.array([[(ci+.5)/len(configs[kind])]+[float.fromhex(x) for x in item['unit_hex']]],dtype=np.float64)
            result['source_face_calls']+=1;save();(fy,fa,fb,tau),face_cpu=timed('face_points',face.face_points,4,kind,U)
            snapshot('active_face',{'index':idx,'U_hex':[[float(v).hex() for v in r] for r in U],'y_hex':[[float(v).hex() for v in r] for r in fy],'a':[int(v) for v in fa],'b':[int(v) for v in fb],'tau_hex':[float(v).hex() for v in tau],'cpu':face_cpu})
            y=[F.from_float(float(v)) for v in fy[0]];assert fa[0]==a and fb[0]==b and y[a]==y[b]
            low=max(y[j] for j in range(4) if j not in (a,b) and j!=top) if kind=='23' else max(y[j] for j in range(4) if j not in (a,b))
            window=min((y[top]-y[a])/3,y[a]-low) if kind=='23' else (y[a]-low)/3
            window=min(window,F.from_float(8.5/math.sqrt(2)));assert window>0
            max_window_difference=max(max_window_difference,abs(math.sqrt(2)*float(window)-float(tau[0])))
            direction=[F(int(j==a)-int(j==b)) for j in range(4)];ledger=Ledger();result['active_point'].update(y=y,window_delta=window,phase='actual_partition',ledger=ledger.counts);result['model_partition_calls']+=1;save()
            (cells,layers),actual_cpu=timed('actual_partition',partition,model,y,direction,-window,window,ledger)
            snapshot('active_actual_partition',{'index':idx,'cells':cells,'layers':layers,'cpu':actual_cpu,'ledger':ledger.counts})
            result['active_point'].update(phase='frozen_partition',actual_layers=layers);result['model_partition_calls']+=1;save();(fcells,flayers),frozen_cpu=timed('frozen_partition',partition,model,y,direction,-window,window,ledger,frozen=True)
            snapshot('active_frozen_partition',{'index':idx,'cells':fcells,'layers':flayers,'cpu':frozen_cpu,'ledger':ledger.counts})
            raw={'index':idx,**item,'y':y,'window_delta':window,'source_tau_z':float(tau[0]),'layers':layers,'frozen_layers':flayers,'halves':[],'cpu':{'face':face_cpu,'actual_partition':actual_cpu,'frozen_partition':frozen_cpu}}
            if idx in full_ids:raw['cells']=cells;raw['frozen_cells']=fcells
            def save_active():
                snapshot('active_point',raw)
            save_active();result['active_point']['phase']='half_intervals';save()
            summary={'cpu':raw['cpu'],'halves':[]}
            for side,lo,hi,truth in [('negative',-window,F(0),b if kind=='23' else a),('positive',F(0),window,a if kind=='23' else b)]:
                partial={'side':side,'truth':truth};raw['pending_half']=partial;save_active()
                actual,ae_cpu=timed('actual_errors',errors,cells,truth,lo,hi,ledger);partial.update(actual_errors=actual,actual_errors_cpu=ae_cpu);save_active()
                frozen,fe_cpu=timed('frozen_errors',errors,fcells,truth,lo,hi,ledger);partial.update(frozen_errors=frozen,frozen_errors_cpu=fe_cpu);save_active()
                bands,match_cpu=timed('matching',signed_difference,actual,frozen,ledger);partial.update(bands=bands,matching_cpu=match_cpu);save_active()
                t=time.process_time();actual_mass=math.fsum(mass(l,r) for l,r in actual);am_cpu=time.process_time()-t;partial.update(actual_mass=actual_mass,actual_mass_cpu=am_cpu);save_active()
                t=time.process_time();frozen_mass=math.fsum(mass(l,r) for l,r in frozen);fm_cpu=time.process_time()-t;partial.update(frozen_mass=frozen_mass,frozen_mass_cpu=fm_cpu);save_active()
                values={};times={};details={};partial.update(correction=values,times=times,terms=details)
                for mode in ['first','second','direct']:
                    (value,terms),dt=timed(mode,evaluate,bands,mode);values[mode]=value;times[mode]=dt;details[mode]=terms;save_active()
                gap=abs((actual_mass-frozen_mass)-values['direct'])
                d=[float(r-l) for l,r,s in bands];bound1=math.exp(-.5)/math.sqrt(2*math.pi)*math.fsum(v*v for v in d);bound2=math.fsum(abs(v)**3 for v in d)/(3*math.sqrt(math.pi))
                h={'side':side,'truth':truth,'actual_errors':actual,'frozen_errors':frozen,'bands':bands,'window_mass':mass(lo,hi),'actual_mass':actual_mass,'frozen_mass':frozen_mass,'correction':values,'terms':details,'bound1':bound1,'bound2':bound2,'parity_error':gap,'cpu':{'actual_errors':ae_cpu,'frozen_errors':fe_cpu,'matching':match_cpu,'actual_mass':am_cpu,'frozen_mass':fm_cpu,**times}}
                raw['halves'].append(h);raw.pop('pending_half');save_active();assert gap<=3e-13,'direct signed-band parity'
                summary['halves'].append({k:h[k] for k in ['truth','window_mass','actual_mass','frozen_mass','correction','bound1','bound2','cpu']})
            raw['ledger']=ledger.counts;raw['returned_bit_max']=[ledger.max_returned_num_bits,ledger.max_returned_den_bits]
            line=json.dumps(encode(raw),separators=(',',':'),allow_nan=False)+'\n';result['raw_bytes']+=len(line.encode());cap()
            with (OUT/'points.jsonl').open('a') as f:f.write(line)
            perpoint.append(summary);result['points']+=1;result['active_point']['phase']='complete';event('point_completed',index=idx)
            if idx==17:
                elapsed=time.process_time()-started;projection=elapsed+(elapsed-model_cpu-table_cpu)*2700/18
                result['resource_projection']={'pilot_points':18,'recorded_cpu':elapsed,'naive_full_projection':projection,'limit':75,'qualification':'Scheduling discriminator only; central nodes need not represent later complexity.'};save()
                if projection>75:raise RuntimeError('predeclared resource projection exceeds75CPU')
        result.pop('active_point',None);result['max_normal_window_difference_from_source']=max_window_difference
        for order in [3,5]:
            strata={(kind,k):{m:[] for m in ['zero','first','second','direct','actual','bound1','bound2','window_mass']} for kind in ['23','12'] for k in range(4)}
            ids=sorted({i for i,w in members[order]});costs={m:model_cpu+table_cpu+rule_cpu[order] for m in ['zero','first','second','direct','actual']}
            for idx in ids:
                r=perpoint[idx];c=r['cpu']
                for m in costs:costs[m]+=c['face']+(c['frozen_partition'] if m=='zero' else c['actual_partition'] if m=='actual' else c['actual_partition']+c['frozen_partition'])
                for h in r['halves']:
                    c=h['cpu'];costs['zero']+=c['frozen_errors']+c['frozen_mass'];costs['actual']+=c['actual_errors']+c['actual_mass']
                    for m in ['first','second','direct']:costs[m]+=c['actual_errors']+c['frozen_errors']+c['matching']+c['frozen_mass']+c[m]
            for idx,w in members[order]:
                for h in perpoint[idx]['halves']:
                    d=strata[(catalogue[idx]['kind'],h['truth'])];d['window_mass'].append(w*h['window_mass']);d['zero'].append(w*h['frozen_mass']);d['actual'].append(w*h['actual_mass'])
                    for m in ['first','second','direct']:d[m].append(w*(h['frozen_mass']+h['correction'][m]))
                    for m in ['bound1','bound2']:d[m].append(w*h[m])
            sums={f'{kind}:{k}':{m:math.fsum(v) for m,v in d.items()} for (kind,k),d in strata.items()};total={m:math.fsum(d[m] for d in sums.values()) for m in next(iter(sums.values()))}
            metrics={}
            for m in costs:
                accuracy=1-total[m];delta=[F.from_float(accuracy)-F.from_float(z) for z in [reflo,refhi]];elo=F(0) if delta[0]*delta[1]<=0 else min(abs(z) for z in delta);ehi=max(abs(z) for z in delta)
                metrics[m]={'accuracy':accuracy,'composed_cpu':costs[m],'absolute_error_interval':[float(elo),float(ehi)],'cpu_squared_error_interval':[float(F.from_float(costs[m])*elo*elo),float(F.from_float(costs[m])*ehi*ehi)]}
            for m in ['first','second','direct','actual']:metrics[m]['robust_halving_vs_zero_on_recorded_times']=metrics[m]['cpu_squared_error_interval'][1]<.5*metrics['zero']['cpu_squared_error_interval'][0]
            result['summary'].append({'order':order,'strata':sums,'totals':total,'metrics':metrics});save()
        assert result['points']==2718 and result['source_face_calls']==2718 and result['model_partition_calls']==5436
        result['points_sha256']=sha(OUT/'points.jsonl');result['qualification']='Fixed single-model CPU diagnostic; no FPE/ARC/HP1 claim. Outer/table/float errors separate;64 full-cell records, no assumed exhaustive independent recurrence review.'
        event('completed');assert result['events']==5439;result['complete']=result['pass_all']=True;save()
    except BaseException:result['failure']=traceback.format_exc();save();raise

if __name__=='__main__':main()
