"""Eight pair-reference domains then six inherited analytic networks only."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations
from collections import Counter
import json,hashlib,time,resource
import mpmath as mp
from primitive import pack,mpq
from network_pairs import unpack,prepare_pair,evaluate_pair
from pair_reference import integrate
HERE=Path(__file__).resolve().parent

def reference_controls(record):
    sector=dict(lo=(F(0),F(-1)),hi=(F(0),F(1)))
    cases=[('positive_z',[0,0,1],[0,0,1],'quarter'),('negative_z',[0,0,-1],[0,0,-1],'quarter'),
           ('opposite',[0,0,1],[0,0,-1],'zero'),('constant_x',[1,0,0],[1,0,0],'half'),
           ('x_z',[1,0,0],[0,0,1],'cross'),('negative_x',[-1,0,0],[0,0,1],'zero'),
           ('bounded',[1,0,1],[1,0,-1],'twocross'),('zero',[0,0,0],[1,0,0],'zero')]
    for name,v,w,kind in cases:
        record(dict(stage='reference_control_attempt',name=name));t=time.process_time()
        value=integrate(sector,tuple(map(F,v)),tuple(map(F,w)));cpu=time.process_time()-t
        with mp.workdps(160):
            expected={'quarter':mp.mpf(1)/4,'half':mp.mpf(1)/2,'zero':mp.mpf(0),'cross':1/(2*mp.pi),'twocross':1/mp.pi}[kind]
            err=abs(mp.mpf(value['value'])-expected)
            yield dict(name=name,v=v,w=w,reference=value,reference_cpu=cpu,expected=mp.nstr(expected,160),absolute_error=mp.nstr(err,100),passed=bool(err<=mp.mpf('1e-50')))

def main():
    dest=HERE/'network_results.json';progress=HERE/'network_progress.jsonl';assert not dest.exists() and not progress.exists()
    P=json.loads((HERE/'network_protocol.json').read_text());assert P['source_gate']
    for path,sha in P['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
    data=json.loads(Path(P['sector_source']).read_text());assert data['complete'] and data['all_checks_pass'] and len(data['rows'])==6
    result=dict(complete=False,reference_controls=[],rows=[],protocol_sha256=hashlib.sha256((HERE/'network_protocol.json').read_bytes()).hexdigest());start=time.process_time()
    def record(x):
        with progress.open('a') as f:f.write(json.dumps(pack(x),allow_nan=False)+'\n')
    def save():
        result.update(cpu=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');tmp.replace(dest)
    try:
        record(dict(stage='reference_controls_attempt',count=8))
        for row in reference_controls(record):
            result['reference_controls'].append(row);record(dict(stage='reference_control',row=row));save();assert row['passed'],row['name']
        for saved in data['rows']:
            name=saved['name'];cert=unpack(saved['certificate']);H=cert['H'];assert H==4
            pairs=[(i,j) for i in range(H) for j in range(i,H)]
            row=dict(name=name,cells=[],complete=False);result['rows'].append(row);record(dict(stage='network_attempt',name=name));save()
            for k,sector in enumerate(cert['sectors']):
                for i,j in pairs:
                    record(dict(stage='pair_attempt',name=name,sector=k,i=i,j=j));v=sector['h3_rows'][i];w=sector['h3_rows'][j]
                    t=time.process_time();c=prepare_pair(sector,v,w);low=evaluate_pair(c,v,w,80);high=evaluate_pair(c,v,w,120);ct=time.process_time()-t
                    t=time.process_time();ref=integrate(sector,v,w);rt=time.process_time()-t
                    cell=dict(sector=k,i=i,j=j,certificate=c,value80=low,value120=high,reference=ref,candidate_cpu=ct,reference_cpu=rt)
                    if high.get('value') is None:cell.update(passed=False,reason='unsupported preserved')
                    else:
                        with mp.workdps(160):
                            x=mp.mpf(high['value']);y=mp.mpf(ref['value']);e=abs(x-y)
                            cell.update(absolute_error=mp.nstr(e,100),passed=bool(e<=mp.mpf('1e-50')+mp.mpf('1e-40')*abs(y) and abs(x-mp.mpf(low['value']))<=mp.mpf('1e-60')*(1+abs(x)) and x>=-mp.mpf('1e-60')))
                    row['cells'].append(cell);record(dict(stage='pair_result',name=name,cell=cell))
                    if not cell['passed']:save();raise AssertionError((name,k,i,j))
                save()
            with mp.workdps(160):
                raw=mp.matrix(H,H);rr=mp.matrix(H,H)
                for i,j in pairs:
                    raw[i,j]=raw[j,i]=sum((mp.mpf(c['value120']['value']) for c in row['cells'] if (c['i'],c['j'])==(i,j)),mp.mpf(0))
                    rr[i,j]=rr[j,i]=sum((mp.mpf(c['reference']['value']) for c in row['cells'] if (c['i'],c['j'])==(i,j)),mp.mpf(0))
                mu=mp.matrix([mp.mpf(v) for v in saved['mean']]);cov=raw-mu*mu.T
                minors=[]
                for size in range(1,H+1):
                    for indices in combinations(range(H),size):minors.append(dict(indices=indices,value=mp.nstr(mp.det(mp.matrix([[cov[i,j] for j in indices] for i in indices])),100)))
                psd=all(mp.mpf(m['value'])>=-mp.mpf('1e-50') for m in minors)
                analytic=[]
                if name in ['zero','zero_recurrent','identity']:
                    for i,j in pairs:
                        if name=='identity' and i!=j:continue
                        if name=='zero':expected=mp.mpf(0)
                        elif name=='zero_recurrent':expected=max(mpq(cert['wi'][i]*cert['wi'][j]),mp.mpf(0))/2
                        else:expected=mpq(cert['wi'][i]**2)*(mp.mpf(3)/2+(1+mp.sqrt(2))/(2*mp.pi))
                        err=abs(raw[i,j]-expected);analytic.append(dict(i=i,j=j,expected=mp.nstr(expected,160),absolute_error=mp.nstr(err,100),passed=bool(err<=mp.mpf('1e-60')*(1+abs(expected)))))
                row.update(raw_second=[[mp.nstr(raw[i,j],120) for j in range(H)] for i in range(H)],reference_raw_second=[[mp.nstr(rr[i,j],100) for j in range(H)] for i in range(H)],
                           covariance=[[mp.nstr(cov[i,j],120) for j in range(H)] for i in range(H)],principal_minors=minors,psd_tolerance_pass=psd,analytic_controls=analytic,
                           complete=True,all_pass=psd and all(x['passed'] for x in analytic) and all(x['passed'] for x in row['cells']))
            record(dict(stage='network_complete',name=name,passed=row['all_pass']));save();assert row['all_pass'],name
        assert sum(len(r['cells']) for r in result['rows'])==340
        result['complete']=True;result['all_pass']=all(r['all_pass'] for r in result['rows'])
    except BaseException as exc:result['failure']=repr(exc);record(dict(stage='failure',exception=repr(exc)));raise
    finally:save()
    print(json.dumps(dict(complete=result['complete'],passed=result['all_pass'],cpu=result['cpu'],cells=sum(len(r['cells']) for r in result['rows']))))
if __name__=='__main__':main()
