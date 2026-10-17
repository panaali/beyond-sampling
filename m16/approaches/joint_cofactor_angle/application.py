"""Original64 stored matrices; no native determinant/source-mass replay."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,json,time,traceback,sys
import numpy as np
import joint
from frozen import rational_bounds as rb
HERE=Path(__file__).resolve().parent

def angle(y,x):
    if x<=0:return None
    return rb.atan_interval(joint.rounded_interval(rb.I.point(y)/x))

def main():
    out=HERE/'application.json';progress=HERE/'application_progress.jsonl';assert not out.exists() and not progress.exists()
    gate=json.loads((HERE/'application_gate.json').read_text())
    for path,pin in gate['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==pin,path
    sys.set_int_max_str_digits(0)
    controls=json.loads((HERE/'controls.json').read_text());retry=json.loads((HERE/'retry_controls.json').read_text())
    assert controls['failures']==['positive','negative','drift'] and controls['cases']==12
    assert retry['complete'] and retry['cases']==3 and retry['failures']==[]
    merged=[r for r in controls['rows'] if 'exception' not in r]+retry['rows']
    assert len(merged)==len({r['label'] for r in merged})==12
    assert {r['label'] for r in retry['rows']}==set(controls['failures'])
    old=json.loads((HERE/'frozen/saved_minors.json').read_text());man=old['manifest']
    assert old['complete'] and len(old['rows'])==64
    for path,pin in man['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==pin,path
    npz=next(path for path in man['pins'] if path.endswith('panel_nodes.npz'))
    expected=[(lab,n,p) for lab in man['labels'] for n in man['nodes'] for p in man['pair_indices']]
    assert [(r['label'],r['node'],r['pair_position']) for r in old['rows']]==expected
    joint.reset();start=time.process_time();rows=[];fail=[];statuses=Counter()
    with np.load(npz) as archive:
        arrays={lab:{key:archive[f'{lab}_{key}'] for key in ('Rt','detRt','atan2_arguments')} for lab in man['labels']}
    for saved in old['rows']:
        lab,node,pos=saved['label'],saved['node'],saved['pair_position'];key=[lab,node,pos]
        with progress.open('a') as f:f.write(json.dumps({'attempt':key})+'\n')
        try:
            arr=arrays[lab];a=[[F.from_float(float(x)) for x in row] for row in arr['Rt'][node]]
            pair=man['pair_order'][pos];got=joint.enclose(a,pair);statuses[got['status']]+=1
            source_y=float(arr['atan2_arguments'][pos,0,node]);source_x=float(arr['atan2_arguments'][pos,1,node]);native_d=float(arr['detRt'][node])
            assert source_y.hex()==saved['saved_atan2_y']
            row={'key':key,'matrix_hex':[[float(x).hex() for x in r] for r in arr['Rt'][node]],'pair':pair,'target':rb.serial(got),
                 'source_x':source_x.hex(),'source_y':source_y.hex(),'native_det4':native_d.hex(),
                 'max_diagonal_drift':str(max(abs(a[i][i]-1) for i in range(4)))}
            if got['status']=='bounded':
                exact_c=F(int(saved['exact_signed'][0]),int(saved['exact_signed'][1]));assert got['cofactor']==exact_c
                source=angle(F.from_float(source_y),F.from_float(source_x));den_only=angle(-exact_c,F.from_float(source_x))
                candidate=angle(-F.from_float(float.fromhex(saved['candidate_signed_point'])),F.from_float(source_x))
                assert source is not None and den_only is not None and candidate is not None
                target=got['angle'];i,j=pair
                row.update(source_exact_argument_angle=rb.serial(source),source_minus_target=rb.serial(source-target),
                           denominator_only_minus_target=rb.serial(den_only-target),candidate_minus_target=rb.serial(candidate-target),
                           native_det_minus_exact=str(F.from_float(native_d)-got['leading'][-1]),
                           source_x_squared_minus_radicand=str(F.from_float(source_x)**2-got['radicand']),
                           exact_complement_minus_unit_diagonal_surrogate=str(got['det2']-(1-a[i][j]**2)))
        except Exception:
            row={'key':key,'exception':traceback.format_exc()};fail.append(key)
        rows.append(row)
        with progress.open('a') as f:f.write(json.dumps(row)+'\n')
    result={'complete':not fail,'rows':rows,'cases':len(rows),'failures':fail,'statuses':dict(statuses),'accounting':joint.accounting(),
            'cpu_seconds':time.process_time()-start,'pins':gate,'native_mass_model_geometry_MC_calls':0,
            'scope':'Exact stored symmetric SPD matrix angles and exact mathematical angles of source/candidate arguments; excludes covariance construction, libm, coefficients, quadrature and total mass. Source x-squared deviation combines det, om and sqrt rounding; no separate original om reconstruction.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('complete','cases','failures','statuses','cpu_seconds')}));assert not fail
if __name__=='__main__':main()
