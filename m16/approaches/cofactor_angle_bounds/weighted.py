"""Frozen eight-cone numerator-only ideal weighted-angle enclosure."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib, importlib.util, json, math, time
import numpy as np
from bound import enclose, pack, serial

HERE = Path(__file__).resolve().parent
BITS = 160
PAIRS = [(i,j) for i in range(4) for j in range(i+1,4)]

def ceiling(term):
    assert isinstance(term,F) and term >= 0
    q,r = divmod(term.numerator << BITS, term.denominator)
    return q + int(r != 0)

def coefficients(rays, grid):
    """Source-identical coefficient dependencies, no angle/mass evaluation."""
    s,t,w = grid
    N = np.empty((4,4))
    for i in range(4):
        O = np.delete(rays,i,0)
        n = np.array([(-1)**q * np.linalg.det(np.delete(O,q,1)) for q in range(4)])
        N[i] = n * (1.0 if n @ rays[i] > 0 else -1.0) / np.linalg.norm(n)
    C = N @ N.T
    oma = {}
    for i,j in PAIRS:
        dm,dp = np.linalg.norm(N[i]-N[j]),np.linalg.norm(N[i]+N[j])
        oma[i,j] = min(dm,dp)**2 / 2
    Rt = C[None] + s[:,None,None] * (np.eye(4)-C)[None]
    oms=[];weights=[]
    for i,j in PAIRS:
        r=C[i,j];ra=min(abs(r),1.0)
        om=(oma[i,j]+ra*s)*(1+ra*t)
        # A newly frozen binary64 effective coefficient. This is NOT the
        # original sequential angle-multiply/divide/reduction arithmetic.
        weight=(w*r/np.sqrt(np.maximum(1e-300,om)))/(4*math.pi**2)
        oms.append(om);weights.append(weight)
    return Rt,np.array(oms),np.array(weights)

def main():
    out=HERE/'weighted_results.json';progress=HERE/'weighted_progress.jsonl'
    assert not out.exists() and not progress.exists()
    P=json.loads((HERE/'weighted_protocol.json').read_text())
    for name,pin in P['pins'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==pin,name
    controls=json.loads((HERE/'weighted_controls.json').read_text())
    assert controls['failures']==[] and controls['cases']==17
    assert hashlib.sha256((HERE/'weighted_controls.json').read_bytes()).hexdigest()==P['completed_controls_sha256']
    old=json.loads((HERE/'frozen/saved_minors_results.json').read_text())
    reuse={(r['label'],r['pair_position'],r['node']):r for r in old['rows']}
    spec=importlib.util.spec_from_file_location('pinned_guard',HERE/'frozen/primitive.py')
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    def record(row):
        with progress.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    start=time.process_time();record(dict(stage='setup_attempt'))
    x,w=np.polynomial.legendre.leggauss(128);v=35.*(x+1)
    grid=(np.exp(-v),1-np.exp(-v),35.*w*np.exp(-v))
    with np.load(P['rays_path']) as a:rays=a['rays'][P['labels']]
    with np.load(P['nodes_path']) as a:
        saved={label:{key:a[f'{label}_{key}'] for key in ['Rt','detRt','atan2_arguments']} for label in P['labels']}
    rows=[];summaries=[];primitive_counts=Counter();angle_counts=Counter();statuses=Counter();new=0;reused=0
    for label,R in zip(P['labels'],rays):
        record(dict(stage='coefficient_attempt',label=label))
        Rt,oms,weights=coefficients(R,grid);a=saved[label]
        assert Rt.tobytes()==a['Rt'].tobytes(),('Rt parity',label)
        denominators=np.sqrt(np.maximum(a['detRt'],0)[None]*oms)
        assert denominators.tobytes()==a['atan2_arguments'][:,1,:].tobytes(),('denominator parity',label)
        assert np.isfinite(weights).all()
        record(dict(stage='coefficient_parity',label=label,Rt=True,denominators=True))
        units=0;unknown=[];max_term=F(0);peak=0
        for pair,(i,j) in enumerate(PAIRS):
            k,l=[q for q in range(4) if q!=i and q!=j]
            ck,cl=[q for q in range(4) if q!=k],[q for q in range(4) if q!=l]
            sign=(-1)**(k+l)
            for node in range(128):
                record(dict(stage='attempt',label=label,pair=pair,node=node))
                key=(label,pair,node)
                if key in reuse:
                    oldrow=reuse[key];lo,hi=[float.fromhex(t) for t in oldrow['interval']]
                    point=float.fromhex(oldrow['candidate_signed_point'])
                    got=dict(status=oldrow['primitive_status'],counts=oldrow['primitive_counts'])
                    signed=(lo,hi,point);reused+=1
                else:
                    got=guard.enclose(Rt[node][ck][:,cl].tolist());new+=1
                    if got['status']=='unknown_nonfinite':signed=None
                    else:
                        lo,hi,p=got['lower'],got['upper'],got['point']
                        signed=(lo,hi,p) if sign>0 else (-hi,-lo,-p)
                primitive_counts.update(got['counts'])
                angle=None;term=None;u=None
                if signed is not None and all(math.isfinite(z) for z in signed):
                    lo,hi,p=signed
                    angle=enclose(F.from_float(float(denominators[pair,node])),F.from_float(-hi),F.from_float(-lo),F.from_float(-p))
                    angle_counts.update(angle['counts'])
                    if angle['bound'] is not None:
                        term=abs(F.from_float(float(weights[pair,node])))*angle['bound'];u=ceiling(term);units+=u
                        max_term=max(max_term,term);peak=max(peak,term.numerator.bit_length(),term.denominator.bit_length())
                status=angle['status'] if angle else 'unknown_nonfinite_interval';statuses[status]+=1
                if u is None:unknown.append([pair,node,status])
                row=dict(label=label,pair=pair,node=node,reused=key in reuse,primitive_status=got['status'],primitive_counts=got['counts'],cofactor_sign=sign,signed_cofactor_interval=[z.hex() for z in signed[:2]] if signed else None,candidate_signed_point=signed[2].hex() if signed else None,x_hex=float(denominators[pair,node]).hex(),weight_hex=float(weights[pair,node]).hex(),angle=pack(angle) if angle else None,weighted_term=serial(term),ceiling_units=str(u) if u is not None else None)
                rows.append(row);record(row)
        summary=dict(label=label,unknown=unknown,conditional_bound=serial(F(units,1<<BITS)) if not unknown else None,known_terms_bound=serial(F(units,1<<BITS)),max_term=serial(max_term),term_peak_bits=peak,rounding_slack_upper=serial(F(768,1<<BITS)))
        summaries.append(summary);record(dict(stage='cone_complete',**summary))
    result=dict(complete=True,cases=len(rows),new_guard_calls=new,reused_guard_cases=reused,rows=rows,cones=summaries,statuses=dict(statuses),primitive_counts_once=dict(primitive_counts),angle_counts=dict(angle_counts),new_setup_3det_calls=128,new_setup_norm4_calls=128,new_setup_dot4_calls=32,new_setup_matrix4_products=8,grid_leggauss128_calls=1,dyadic_bits=BITS,contraction_named_per_bounded_term=dict(rational_abs=1,rational_mul=1,integer_shift=1,integer_divmod=1,comparison=1,integer_add=2),wrapper_named_counts=dict(numerator_sign_negations=3*len(rows),cofactor_sign_negations=3*sum(r['cofactor_sign']<0 for r in rows),input_dyadic_conversions=5*len(rows),max_term_diagnostic_comparisons=len(rows)),cost_qualifications='Tables retain selected named categories only; bit-length bookkeeping, validation, serialization, interpreter/control/memory and native setup internals not fully priced. Reused64 primitive categories counted once for an ideal total; actual new execution6080.',cpu_seconds=time.process_time()-start,mass_model_geometry_MC_calls=0,scope=P['scope'])
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['cases','new_guard_calls','reused_guard_cases','statuses','cpu_seconds']}))
if __name__=='__main__':main()
