"""Pinned outer3D construction; no source mass or model imports."""
from pathlib import Path
import importlib.util
import hashlib,json,math,time
from collections import Counter
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('positive_inner5_frozen',HERE.parent/'primitive.py')
inner=importlib.util.module_from_spec(spec);spec.loader.exec_module(inner)

def load():
    p=json.loads((HERE/'protocol.json').read_text())
    for name,sha in p['pins'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    return p

def rules(p):
    """Reuse three model-independent source rules; named mapping3n is charged."""
    source=json.loads(Path(p['node_source']).read_text());result={}
    for row in source['orders']:
        n=row['n'];assert n in p['orders']
        result[n]=([(.5*(float.fromhex(x)+1)) for x in row['source_nodes_hex']],
                   [.5*float.fromhex(w) for w in row['source_weights_hex']])
    assert sorted(result)==p['orders']
    return result,{'named_arithmetic':3*sum(p['orders']),
                   'universal_source_leggauss_generation_calls':3,
                   'generation_status':'saved once; original source generation CPU/call metadata pinned, native generation unpriced',
                   'allocation_copy_decode_control':'unpriced'}

def fixtures(p):
    originals=json.loads(Path(p['original_source']).read_text());children=json.loads(Path(p['child_source']).read_text())
    rows=[]
    for r in originals['fixtures']:
        rows.append({'id':'original/'+r['name'],'fixture':r['name'],'kind':'original','rays_hex':r['rays_hex'],'reference':r['reference120']})
    for r in children['children']:
        rows.append({'id':'split/'+r['fixture']+'/'+str(r['side']),'fixture':r['fixture'],'kind':'split','side':r['side'],'rays_hex':r['rays_hex'],'reference':r['reference120']})
    assert len(rows)==33 and len({r['id'] for r in rows})==33
    assert [r['id'] for r in rows]==p['fixture_ids']
    return rows

def one(item,n,rule):
    start=time.process_time();L=inner.Ledger();o=L.op;inner_counts=Counter();xs,ws=rule
    # Global fixed permutation, never chosen from observed values.
    raw=[[float.fromhex(x) for x in row] for row in item['rays_hex']]
    R=[[row[j] for j in [2,3,4,0,1]] for row in raw]
    # The real determinant magnitude is permutation invariant. Its actual
    # installed 5x5 kernel and rounding remain opaque here, not counted as1 flop.
    det=o('abs',float(np.linalg.det(np.array(R,dtype=np.float64))))
    factor=o('mul',det,o('div',3.0,o('mul',8.0,o('mul',math.pi,math.pi))))
    total=0.0;calls=0;branches=Counter();failure=None
    for i,u in enumerate(xs):
        for j,v in enumerate(xs):
            for k,s in enumerate(xs):
                a=o('sub',1.0,u);b=o('sub',1.0,v);c=o('sub',1.0,s)
                ab=o('mul',a,b);l1=o('mul',a,v);l2=o('mul',ab,s);l3=o('mul',ab,c)
                jac=o('mul',o('mul',o('mul',a,a),a),o('mul',o('mul',b,b),c))
                P=[L.total([o('mul',u,row[0]),o('mul',l1,row[1]),o('mul',l2,row[2]),o('mul',l3,row[4])]) for row in R]
                D=[o('mul',l3,o('sub',row[3],row[4])) for row in R]
                ans=inner.integrate(P,D);calls+=1;inner_counts.update(ans['counts']);branches[ans['branch'] or 'none']+=1
                if ans['status']!='ok':
                    failure={'node':[i,j,k],'P_hex':[x.hex() for x in P],'D_hex':[x.hex() for x in D],'inner_result':ans};break
                term=o('mul',o('mul',o('mul',ws[i],ws[j]),ws[k]),o('mul',jac,ans['value']))
                total=o('add',total,term)
            if failure:break
        if failure:break
    result={'id':item['id'],'fixture':item['fixture'],'kind':item['kind'],'order':n,'reference':item['reference'],
            'inner_calls':calls,'intended_inner_calls':n**3,'branches':dict(branches),
            'construction_assembly_counts':L.counts,'inner_counts':dict(inner_counts),
            'opaque_native_calls':{'det_float64_5x5':1},'determinant_hex':det.hex(),
            'cpu':time.process_time()-start,'complete':failure is None}
    if 'side' in item:result['side']=item['side']
    if failure:result['failure']=failure
    else:
        value=o('mul',factor,total)
        valid=o('and',o('finite_check',value),o('gt',value,0.0))
        result.update(value=value,value_hex=value.hex(),finite_positive=valid)
    result['cpu']=time.process_time()-start
    return result
