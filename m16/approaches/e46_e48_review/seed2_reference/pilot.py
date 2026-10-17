"""Fixed64 positive transfer control; no geometry, model or source-mass replay."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import hashlib,importlib.util,itertools,json,math,time,resource
import numpy as np
P=Path(__file__).resolve().parent
SCALE=1/(2*math.pi**2)  # Universal constant; floating rounding remains excluded.
class Budget(Exception):pass
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def validate(R,cert,rescue):
    """Independent certificate proof check, not the support-enumeration solver."""
    tick=time.process_time();tracker=rescue.Tracker();F=rescue.Tracked;F.current=tracker
    A=[[F(float(v)) for v in row] for row in R]
    G=[[sum(A[i][k]*A[j][k] for k in range(4)) for j in range(4)] for i in range(4)]
    det=F(0)
    for perm in itertools.permutations(range(4)):
        term=F(1)
        for i in range(4):term=term*A[i][perm[i]]
        inversions=sum(perm[i]>perm[j] for i in range(4) for j in range(i+1,4))
        det=det+(-term if inversions%2 else term)
    assert det==F(cert['det']) and bool(det)
    y=list(map(F,cert['y']));a=F(cert['a']);b=F(cert['b'])
    assert len(y)==4 and min(y)>=0 and sum(y)==1 and 0<a<=b
    gy=[sum(G[i][j]*y[j] for j in range(4)) for i in range(4)]
    slack=[z-a for z in gy]
    assert min(slack)>=0 and all(y[i]*slack[i]==0 for i in range(4))
    assert all(slack[i]==F(cert['slack'][i]) for i in range(4))
    assert sum(y[i]*gy[i] for i in range(4))==a and max(G[i][i] for i in range(4))==b
    return dict(passed=True,exact_inventory=tracker.report(),cpu=time.process_time()-tick,integer_permutation_and_loop_work_separate=True)

def main():
    start=time.process_time();proto=json.loads((P/'pilot_protocol.json').read_text())
    assert proto['capture_review'] and proto['source_gate']
    paths=[P/'pilot.json',P/'pilot_certificates.json',P/'pilot_progress.jsonl']
    assert not any(p.exists() for p in paths)
    for p,h in proto['pins'].items():assert sha(p)==h,p
    out=dict(complete=False,rows=[],protocol_sha256=sha(P/'pilot_protocol.json'),geometry_evaluations=0,model_input_evaluations=0,source_mass_evaluations=0,reference_order=32)
    certificates=dict(complete=False,rows=[],protocol_sha256=out['protocol_sha256'])
    rules=None;ref_rules=None
    def log(row):
        with paths[2].open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    def save():
        out.update(cpu=time.process_time()-start,process_cpu=time.process_time(),maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if rules is not None:out.update(candidate_rule_events=rules.events,candidate_rule_counts=rules.ledger.dict())
        if ref_rules is not None:out.update(reference_rule_events=ref_rules.events,reference_rule_arithmetic=ref_rules.node_plan_arithmetic)
        write(paths[0],out)
    def limit():
        if time.process_time()-start>18:raise Budget('18CPU soft cap; no automatic resume')
    try:
        log(dict(stage='load_attempt'));capture=json.loads((P/'capture.json').read_text());assert capture['complete'] and capture['N']==proto['N']
        arrays=[]
        for r in capture['parents']:
            path=P/'parents'/r['archive'];assert sha(path)==r['sha256']
            with np.load(path,allow_pickle=False) as z:
                assert z.files==['rays'];a=z['rays']
            assert a.shape==(r['N'],4,4) and a.dtype==np.float64 and np.isfinite(a).all();arrays.append(a)
        rays=np.concatenate(arrays);assert rays.shape==(proto['N'],4,4)
        assert hashlib.sha256(rays.tobytes(order='C')).hexdigest()==proto['ordered_ray_sha256']
        indices=proto['indices'];assert indices==[i*(len(rays)-1)//63 for i in range(64)] and len(set(indices))==64
        spec=importlib.util.spec_from_file_location('seed2_pinned_adapter',proto['adapter']);adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        unused_engine,unused_guards,M,old,kernel,rescue,radial=adapter.load()
        target=Fraction(1,10**7*len(rays));M.TOL=1e-7/len(rays);down=Fraction(M.TOL)>target
        if down:M.TOL=math.nextafter(M.TOL,0.)
        assert Fraction(M.TOL)<=target;rescue.TOLERANCE=target
        rules=kernel.Rules();ref_rules=old.Rules()
        out.update(N=len(rays),indices=indices,allocation=dict(exact=str(target),floating=M.TOL,nextafter_down=down),loaded_and_concatenated_float64_elements=2*len(rays)*16,load_and_import_cpu=time.process_time()-start)
        save()
        # Freeze and independently validate every exact certificate before any mass call.
        for index in indices:
            limit();log(dict(stage='certificate_attempt',index=index));tick=time.process_time()
            tracker=rescue.Tracker();rescue.Tracked.current=tracker;cert=radial.bounds(rays[index]);construction_cpu=time.process_time()-tick;inventory=tracker.report()
            row=dict(index=index,certificate=cert,construction_cpu=construction_cpu,construction_inventory=inventory)
            certificates['rows'].append(row);write(paths[1],certificates)
            row['independent_validation']=validate(rays[index],cert,rescue);write(paths[1],certificates)
            log(dict(stage='certificate_complete',index=index));save()
        certificates['complete']=True;write(paths[1],certificates);out['certificates_sha256']=sha(paths[1]);save()
        for prepared in certificates['rows']:
            limit();index=prepared['index'];log(dict(stage='mass_attempt',index=index));R=rays[index]
            row=dict(index=index);out['rows'].append(row)
            row['candidate']=M.one(R,old,kernel,rescue,radial,rules);save()
            assert row['candidate']['mode']!='failure',row['candidate']
            cert=prepared['certificate'];det=float(abs(Fraction(cert['det'])));assert math.isfinite(det) and det>0
            tick=time.process_time();bd=rescue.bound(cert,det,32,dps=80)
            row['reference_bound']=dict(**bd,ordinary_mp_conditional=True,cpu=time.process_time()-tick)
            assert math.isfinite(bd['total_bound']) and bd['total_bound']>=0
            tick=time.process_time();y,w=ref_rules.get(32);points=y@R
            value=float((det*SCALE)*np.dot(w,1/np.sum(points*points,axis=1)**2))
            row['reference']=dict(value=value,determinant_float=det,kernel_arithmetic=39*32**3+1,cpu_including_rule_setup=time.process_time()-tick,matmul_reduction_native_internals_unpriced=True)
            assert math.isfinite(value) and value>0
            candidate=row['candidate']
            if candidate['mode']=='returned':
                row['bound_pass']=candidate['bound']<=M.TOL
                row['difference']=candidate['value']-value
                row['comparison_pass']=abs(row['difference'])<=candidate['bound']+math.nextafter(bd['total_bound'],math.inf)+5e-14
            save();log(dict(stage='mass_complete',index=index,mode=candidate['mode']))
            assert row.get('bound_pass',True) and row.get('comparison_pass',True),row
        assert sha(paths[1])==out['certificates_sha256']
        known=Counter()
        for row in out['rows']:
            c=row['candidate'];known.update(c['center']['setup_selection_counts']);known.update(c.get('repeated_determinant_counts',{}));known.update(c.get('kernel_counts',{}))
        known.update(rules.ledger.dict());known['universal_constant_arithmetic']+=6
        good=[r for r in out['rows'] if r['candidate']['mode']=='returned']
        out.update(complete=True,returned=len(good),unresolved_indices=[r['index'] for r in out['rows'] if r['candidate']['mode']!='returned'],methods=dict(Counter(r['candidate'].get('method',r['candidate']['mode']) for r in out['rows'])),known_candidate_counts=dict(known),known_candidate_total=sum(known.values()),reference_kernel_arithmetic=sum(r['reference']['kernel_arithmetic'] for r in out['rows']),reference_universal_constant_arithmetic=3,adapter_cold_setup_outside_known_candidate_total=True,covered_candidate_probability=math.fsum(r['candidate']['value'] for r in good),covered_candidate_bound=math.fsum(r['candidate']['bound'] for r in good),difference_l1=math.fsum(abs(r['difference']) for r in good),maximum_absolute_difference=max((abs(r['difference']) for r in good),default=0),reference_loose_bounds=sum(r['reference_bound']['total_bound']>M.TOL for r in out['rows']),full_floating_certificate=False,full_cost=False)
        log(dict(stage='complete',returned=len(good)));save()
    except BaseException as e:
        out['failure']=repr(e);log(dict(stage='failure',error=repr(e)));save();raise
    print(json.dumps({k:out[k] for k in ('complete','returned','unresolved_indices','known_candidate_total','difference_l1','reference_loose_bounds','cpu')}))

if __name__=='__main__':main()
