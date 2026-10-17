"""66 fixed split-rule rows, reviewed pilot reused once; no old mass replay."""
from pathlib import Path
import hashlib,json,math,time
from collections import Counter
import numpy as np
from primitive import integrate
from outer import construct,scalar_construction,safe,H

def main():
    out=H/'full.json';progress=H/'full_progress.jsonl';dest=H/'full_arrays';assert not out.exists() and not progress.exists() and not dest.exists()
    def record(r):
        with progress.open('a') as f:f.write(json.dumps(safe(r),allow_nan=False)+'\n')
    record(dict(stage='pins_attempt'));protocol=json.loads((H/'full_protocol.json').read_text())
    for p,h in protocol['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    pilot=json.loads((H/'outer.json').read_text());assert pilot['all_pass'];reused=next(r for r in pilot['rows'] if r['id']=='split/positive5orthant/0')
    archive=H/'outer_case_1.npz';assert hashlib.sha256(archive.read_bytes()).hexdigest()==reused['archive_sha256']
    start=time.process_time();dest.mkdir();node_source=json.loads(Path(protocol['node_source']).read_text());rules={}
    for r in node_source['orders']:
        n=r['n'];assert n in [6,10,16]
        rules[n]=([.5*(float.fromhex(z)+1) for z in r['source_nodes_hex']],[.5*float.fromhex(z) for z in r['source_weights_hex']])
    assert sorted(rules)==[6,10,16]
    children=json.loads(Path(protocol['child_source']).read_text())['children'];assert len(children)==22
    saved=json.loads(Path(protocol['saved_full']).read_text());prior={(r['id'],r['order']):r for r in saved['rows'] if r['kind']=='split'};assert len(prior)==66
    rows=[];new_calls=0;reuses=0
    for ci,item in enumerate(children):
      ident='split/'+item['fixture']+'/'+str(item['side'])
      for n in [6,10,16]:
        record(dict(stage='case_attempt',id=ident,order=n));old=prior[ident,n]
        if ident==reused['id'] and n==6:
            row=dict(reused);row.update(reused_pilot=True,archive=str(archive),reference=old['reference']);rows.append(row);reuses+=1;record(dict(stage='case_result',row=row));continue
        t=time.process_time();archive=dest/f'case_{ci:02d}_{n}.npz';assert not archive.exists();x,w=rules[n]
        raw=[[float.fromhex(z) for z in r] for r in item['rays_hex']];R=[[raw[i][j] for i in [2,3,4,0,1]] for j in range(5)]
        P,D,jac,weights=construct(np.array(R),np.array(x),np.array(w));reference=scalar_construction(R,x,w)
        checks={name:a.tobytes()==b.tobytes() for name,a,b in zip(['P','D','jacobian','weights'],[P,D,jac,weights],reference,strict=True)}
        record(dict(stage='construction_result',id=ident,order=n,checks=checks));assert all(checks.values())
        det=abs(float(np.linalg.det(np.array(R,dtype=np.float64))));factor=det*(3.0/(8.0*(math.pi*math.pi)))
        ans=integrate(P,D);assert len(ans)==n**3;new_calls+=len(ans)
        bad=[dict(node=i,result=r) for i,r in enumerate(ans) if r['status']!='ok'];branches=dict(Counter(r['branch'] or 'none' for r in ans))
        record(dict(stage='inner_result',id=ident,order=n,failures=bad,branches=branches));assert not bad
        values=np.array([r['value'] for r in ans]);total=0.0
        for ww,jj,vv in zip(weights,jac,values,strict=True):total=total+float(ww)*(float(jj)*float(vv))
        value=factor*total
        np.savez(archive,P=P,D=D,jacobian=jac,weights=weights,inner_values=values,R=np.array(R),nodes=np.array(x),rule_weights=np.array(w))
        parity=dict(value=value.hex()==old['value_hex'],determinant=det.hex()==old['determinant_hex'],branches=branches==old['branches'],inner_calls=old['inner_calls']==n**3)
        row=dict(id=ident,order=n,value_hex=value.hex(),determinant_hex=det.hex(),saved_value_hex=old['value_hex'],saved_determinant_hex=old['determinant_hex'],branches=branches,construction_parity=checks,parity=parity,passed=all(parity.values()),inner_calls=n**3,cpu_seconds=time.process_time()-t,archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),reused_pilot=False,reference=old['reference'])
        rows.append(row);record(dict(stage='case_result',row=row))
    assert len(rows)==66 and reuses==1 and new_calls==116648
    result=dict(complete=True,rows=rows,all_pass=all(r['passed'] for r in rows),reused_pilot_rows=1,new_inner_calls=new_calls,total_split_inner_calls=sum(r['inner_calls'] for r in rows),new_mass_evaluations=65,new_det5_calls=65,new_scalar_construction_nodes=new_calls,new_mapped_constant_operations=96,canonical_mapping_operations=96,actually_executed_staged_mapping_operations=114,original_pilot_extra_inner_calls=216,pilot_cpu_seconds=pilot['cpu_seconds'],cpu_seconds=time.process_time()-start,old_scalar_inner_reference_geometry_model_MC_calls=0,scope=protocol['scope'])
    out.write_text(json.dumps(safe(result),indent=2,allow_nan=False)+'\n');print(json.dumps(dict(all_pass=result['all_pass'],new_inner_calls=new_calls,cpu_seconds=result['cpu_seconds'])));assert result['all_pass'],'durable saved parity failure'
if __name__=='__main__':
    try:main()
    except BaseException as e:
        with (H/'full_failures.jsonl').open('a') as f:f.write(json.dumps(dict(type=type(e).__name__,message=str(e)))+'\n')
        raise
