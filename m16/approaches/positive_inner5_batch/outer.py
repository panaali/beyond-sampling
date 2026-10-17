"""Frozen two-case n6 construction/assembly pilot; no old inner execution."""
from pathlib import Path
import hashlib,json,math,time
from collections import Counter
import numpy as np
from primitive import integrate
H=Path(__file__).resolve().parent

def safe(x):
    if isinstance(x,float) and not math.isfinite(x):return x.hex()
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,list):return [safe(v) for v in x]
    return x

def construct(R,x,w):
    u,v,s=np.meshgrid(x,x,x,indexing='ij');u=u.ravel();v=v.ravel();s=s.ravel()
    wi,wj,wk=np.meshgrid(w,w,w,indexing='ij');wi=wi.ravel();wj=wj.ravel();wk=wk.ravel()
    a=1.0-u;b=1.0-v;c=1.0-s;ab=a*b;l1=a*v;l2=ab*s;l3=ab*c
    jac=((a*a)*a)*((b*b)*c)
    P=u[:,None]*R[:,0];P=P+l1[:,None]*R[:,1];P=P+l2[:,None]*R[:,2];P=P+l3[:,None]*R[:,4]
    D=l3[:,None]*(R[:,3]-R[:,4]);weights=(wi*wj)*wk
    return P,D,jac,weights

def scalar_construction(R,x,w):
    # Independent nested traversal, Python scalar arithmetic only. No integral.
    ps=[];ds=[];js=[];ws=[]
    for i,u in enumerate(x):
      for j,v in enumerate(x):
       for k,s in enumerate(x):
        a=1.0-u;b=1.0-v;c=1.0-s;ab=a*b;l1=a*v;l2=ab*s;l3=ab*c
        jac=((a*a)*a)*((b*b)*c);p=[];d=[]
        for row in R:
            q=u*row[0];q=q+l1*row[1];q=q+l2*row[2];q=q+l3*row[4]
            p.append(q);d.append(l3*(row[3]-row[4]))
        ps.append(p);ds.append(d);js.append(jac);ws.append((w[i]*w[j])*w[k])
    return tuple(np.array(v,dtype=np.float64) for v in (ps,ds,js,ws))

def main():
    out=H/'outer.json';progress=H/'outer_progress.jsonl';assert not out.exists() and not progress.exists()
    def record(r):
        with progress.open('a') as f:f.write(json.dumps(safe(r),allow_nan=False)+'\n')
    record(dict(stage='pins_attempt'));protocol=json.loads((H/'outer_protocol.json').read_text())
    for p,h in protocol['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    controls=json.loads((H/'controls.json').read_text());assert controls['all_pass'] and controls['cases']==30
    start=time.process_time();nodes=json.loads(Path(protocol['node_source']).read_text());nr=next(r for r in nodes['orders'] if r['n']==6)
    x=[.5*(float.fromhex(z)+1) for z in nr['source_nodes_hex']];w=[.5*float.fromhex(z) for z in nr['source_weights_hex']]
    originals=json.loads(Path(protocol['original_source']).read_text());children=json.loads(Path(protocol['child_source']).read_text());saved=json.loads(Path(protocol['saved_full']).read_text())
    items=[next(r for r in originals['fixtures'] if r['name']=='positive5orthant'),next(r for r in children['children'] if r['fixture']=='positive5orthant' and r['side']==0)]
    rows=[]
    for index,(ident,item) in enumerate(zip(protocol['ids'],items,strict=True)):
        archive=H/f'outer_case_{index}.npz';assert not archive.exists();record(dict(stage='case_attempt',id=ident));t=time.process_time()
        raw=[[float.fromhex(z) for z in row] for row in item['rays_hex']];R=[[raw[i][j] for i in [2,3,4,0,1]] for j in range(5)]
        if index==1:assert raw!=list(map(list,zip(*raw)))
        P,D,jac,weights=construct(np.array(R),np.array(x),np.array(w));ref=scalar_construction(R,x,w)
        checks={name:a.tobytes()==b.tobytes() for name,a,b in zip(['P','D','jacobian','weights'],[P,D,jac,weights],ref,strict=True)}
        record(dict(stage='construction_result',id=ident,checks=checks));assert all(checks.values())
        det=abs(float(np.linalg.det(np.array(R,dtype=np.float64))));factor=det*(3.0/(8.0*(math.pi*math.pi)))
        ans=integrate(P,D);assert len(ans)==216
        bad=[dict(node=i,result=r) for i,r in enumerate(ans) if r['status']!='ok'];branches=dict(Counter(r['branch'] or 'none' for r in ans))
        record(dict(stage='inner_result',id=ident,failures=bad,branches=branches));assert not bad
        values=np.array([r['value'] for r in ans]);total=0.0
        for ww,jj,vv in zip(weights,jac,values,strict=True):total=total+float(ww)*(float(jj)*float(vv))
        value=factor*total;prior=next(r for r in saved['rows'] if r['id']==ident and r['order']==6)
        np.savez(archive,P=P,D=D,jacobian=jac,weights=weights,inner_values=values,R=np.array(R),nodes=np.array(x),rule_weights=np.array(w))
        parity=dict(value=value.hex()==prior['value_hex'],determinant=det.hex()==prior['determinant_hex'],branches=branches==prior['branches'],inner_calls=prior['inner_calls']==216)
        row=dict(id=ident,order=6,value_hex=value.hex(),determinant_hex=det.hex(),saved_value_hex=prior['value_hex'],saved_determinant_hex=prior['determinant_hex'],branches=branches,construction_parity=checks,parity=parity,passed=all(parity.values()),inner_calls=216,cpu_seconds=time.process_time()-t,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        rows.append(row);record(dict(stage='case_result',row=row))
    result=dict(complete=True,rows=rows,all_pass=all(r['passed'] for r in rows),new_inner_calls=432,new_det5_calls=2,new_mass_evaluations=2,old_scalar_inner_reference_geometry_model_MC_calls=0,mapped_constant_operations=18,inherited_source_rules='saved n6 generation reused once; all prior setup provenance retained',diagnostic_scalar_construction_nodes=432,cpu_seconds=time.process_time()-start,scope=protocol['scope'])
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(all_pass=result['all_pass'],cpu_seconds=result['cpu_seconds'])));assert result['all_pass'],'durable parity failure'
if __name__=='__main__':
    try:main()
    except BaseException as e:
        with (H/'outer_failures.jsonl').open('a') as f:f.write(json.dumps(dict(type=type(e).__name__,message=str(e)))+'\n')
        raise
