"""Twelve frozen analytic cases; exact target, MP diagnostics only."""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import hashlib,json,time,traceback
import mpmath as mp
import joint
from frozen import rational_bounds as rb
HERE=Path(__file__).resolve().parent

def identity():return [[F(i==j) for j in range(4)] for i in range(4)]
def cases():
    out=[(f'identity_{i}{j}',identity(),[i,j],'bounded') for i,j in combinations(range(4),2)]
    for label,rho in [('positive',F(1,2)),('negative',F(-1,2)),('drift',F(1,2)),('singular',F(1)),('indefinite',F(2))]:
        a=identity();a[2][3]=a[3][2]=rho
        if label=='drift':a[0][0]+=F(1,1<<52)
        out.append((label,a,[0,1],'unknown_not_spd' if label in ('singular','indefinite') else 'bounded'))
    a=identity();a[0][1]=F(1,4);out.append(('asymmetric',a,[0,1],'unknown_asymmetric'))
    return out

def main():
    out=HERE/'controls.json';progress=HERE/'controls_progress.jsonl';assert not out.exists() and not progress.exists()
    pins=json.loads((HERE/'control_pins.json').read_text())
    for name,pin in pins.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==pin,name
    mp.mp.dps=100;joint.reset();start=time.process_time();rows=[];fail=[]
    for label,a,pair,status in cases():
        with progress.open('a') as f:f.write(json.dumps({'attempt':label})+'\n')
        try:
            got=joint.enclose(a,pair);assert got['status']==status
            diagnostic=None
            if status=='bounded':
                A=mp.matrix([[mp.mpf(x.numerator)/x.denominator for x in r] for r in a]);inv=A**-1
                k,l=[n for n in range(4) if n not in pair]
                corr=-inv[k,l]/mp.sqrt(inv[k,k]*inv[l,l]);ref=mp.asin(corr)
                bound=got['angle'];lo=mp.mpf(bound.lo.numerator)/bound.lo.denominator;hi=mp.mpf(bound.hi.numerator)/bound.hi.denominator
                assert lo-mp.mpf('1e-90')<=ref<=hi+mp.mpf('1e-90')
                if label.startswith('identity'):assert bound.lo==bound.hi==0
                diagnostic={'inverse_asin_reference':mp.nstr(ref,90),'inside_diagnostic':True}
            row={'label':label,'matrix':rb.serial(a),'pair':pair,'expected':status,'result':rb.serial(got),'reference':diagnostic}
        except Exception:
            row={'label':label,'exception':traceback.format_exc()};fail.append(label)
        rows.append(row)
        with progress.open('a') as f:f.write(json.dumps(row)+'\n')
    result={'complete':not fail,'cases':len(rows),'failures':fail,'rows':rows,'accounting':joint.accounting(),'cpu_seconds':time.process_time()-start,'pins':pins,'native_mass_model_MC_calls':0,'scope':'Exact matrix-angle controls; inverse MP diagnostic is not the proof of enclosure.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('complete','cases','failures','cpu_seconds')}))
    assert not fail
if __name__=='__main__':main()
