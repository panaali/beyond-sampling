"""Six inherited analytic network controls, no trained checkpoint or MC."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time,resource
import mpmath as mp
from primitive import prepare,evaluate,mpq,pack
from network import build,check_maps
HERE=Path(__file__).resolve().parent

def reference(sector,v):
    with mp.workdps(80):
        lo=mp.atan2(mpq(sector['lo'][1]),mpq(sector['lo'][0]));hi=mp.atan2(mpq(sector['hi'][1]),mpq(sector['hi'][0]))
        assert -mp.pi/2<=lo<hi<=mp.pi/2
        vv=[mpq(x) for x in v];b=abs(vv[2]);calls=0
        if not any(vv):return {'value':'0.0','evaluations':0}
        knots=[lo,hi]
        if vv[0] or vv[1]:
            z=mp.atan2(-vv[0],vv[1]);knots+= [z+k*mp.pi for k in range(-3,4) if lo<z+k*mp.pi<hi]
        knots=sorted(set(knots))
        def f(t):
            nonlocal calls
            calls+=1;a=vv[0]*mp.cos(t)+vv[1]*mp.sin(t)
            return max(a,mp.mpf(0))*mp.sqrt(mp.pi/2) if b==0 else (b+a*(mp.pi/2+mp.atan(a/b)))/mp.sqrt(2*mp.pi)
        value=mp.quad(f,knots)/(2*mp.pi)
        return {'value':mp.nstr(value,80),'evaluations':calls,'lo':mp.nstr(lo,80),'hi':mp.nstr(hi,80)}

def main():
    dest=HERE/'network_results.json';progress=HERE/'network_progress.jsonl';assert not dest.exists() and not progress.exists()
    p=json.loads((HERE/'network_protocol.json').read_text())
    for name,sha in p['pins'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    def record(x):
        with progress.open('a') as f:f.write(json.dumps(pack(x),allow_nan=False)+'\n')
    start=time.process_time();out={'complete':False,'rows':[],'protocol_sha256':hashlib.sha256((HERE/'network_protocol.json').read_bytes()).hexdigest()}
    def save():
        out.update(cpu=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(pack(out),indent=2,allow_nan=False)+'\n');tmp.replace(dest)
    try:
        for item in p['controls']:
            record({'stage':'network_attempt','name':item['name']});wi=[F(float.fromhex(x)) for x in item['wi_hex']];wh=[[F(float.fromhex(x)) for x in row] for row in item['wh_hex']]
            cert=build(wi,wh);row={'name':item['name'],'certificate':cert,'cells':[],'complete':False};out['rows'].append(row);save()
            maps=check_maps(cert);row['map_controls']=maps;record({'stage':'maps','name':item['name'],'result':maps});save()
            for k,sector in enumerate(cert['sectors']):
                for j,v in enumerate(sector['h3_rows']):
                    record({'stage':'cell_attempt','name':item['name'],'sector':k,'unit':j})
                    c=prepare(sector['normals'],v);v160=evaluate(c,160);v200=evaluate(c,200);ref=reference(sector,v)
                    cell={'sector':k,'unit':j,'facet':c,'value160':v160,'value200':v200,'reference':ref}
                    with mp.workdps(220):
                        x=mp.mpf(v200['value']);y=mp.mpf(ref['value']);delta=abs(x-y)
                        cell.update(absolute_error=mp.nstr(delta,100),reference_pass=bool(delta<=mp.mpf('1e-60')+abs(y)*mp.mpf('1e-50')),
                                    precision_pass=bool(abs(x-mp.mpf(v160['value']))<=mp.mpf('1e-100')*(1+abs(x))))
                    row['cells'].append(cell);record({'stage':'cell_result','name':item['name'],'cell':cell});save()
            with mp.workdps(220):
                means=[sum((mp.mpf(c['value200']['value']) for c in row['cells'] if c['unit']==j),mp.mpf(0)) for j in range(cert['H'])]
                refs=[sum((mp.mpf(c['reference']['value']) for c in row['cells'] if c['unit']==j),mp.mpf(0)) for j in range(cert['H'])]
                mass=sum((mp.atan2(mpq(s['hi'][1]),mpq(s['hi'][0]))-mp.atan2(mpq(s['lo'][1]),mpq(s['lo'][0])) for s in cert['sectors']),mp.mpf(0))/(2*mp.pi)
                row.update(mean=[mp.nstr(v,200) for v in means],reference_mean=[mp.nstr(v,100) for v in refs],sector_mass=mp.nstr(mass,100),mass_pass=bool(abs(mass-1)<mp.mpf('1e-100')))
                if item['name'] in ['zero','zero_recurrent','identity']:
                    scale=mp.mpf(0) if item['name']=='zero' else 1/mp.sqrt(2*mp.pi)
                    if item['name']=='identity':scale*=1+1/mp.sqrt(2)+1/mp.sqrt(3)
                    expected=[abs(mpq(w))*scale for w in wi]
                    row['analytic_mean']=[mp.nstr(v,200) for v in expected]
                    row['analytic_pass']=all(abs(x-y)<=mp.mpf('1e-100')*(1+abs(y)) for x,y in zip(means,expected))
            row['complete']=True;row['all_checks_pass']=row['mass_pass'] and row.get('analytic_pass',True) and all(c['reference_pass'] and c['precision_pass'] for c in row['cells']);record({'stage':'network_complete','name':item['name'],'pass':row['all_checks_pass']});save()
        out['complete']=True;out['all_checks_pass']=all(r['all_checks_pass'] for r in out['rows'])
    except BaseException as exc:
        out['failure']=repr(exc);record({'stage':'failure','exception':repr(exc)});raise
    finally:save()
    print(json.dumps({'complete':out['complete'],'pass':out['all_checks_pass'],'networks':len(out['rows']),'cpu':out['cpu']}))

if __name__=='__main__':main()
