import time
START=time.process_time()
import json,itertools,math,resource
import numpy as np
import audit as A

def main():
 runner=A.Run();old=json.loads((A.HERE.parent/'controls.json').read_text());rows=[];arrays={}
 Cs=[np.eye(4)]+[(.5*np.eye(4)+.5*np.ones((4,4)))*np.outer(s,s) for s in itertools.product([-1,1],repeat=4)]
 cases=[(np.linalg.inv(np.linalg.cholesky(C)).T,r['label'],r['reference'],r['reference_bound'],'ordinary') for C,r in zip(Cs,old['rows'])]
 pins=json.loads((A.HERE.parent/'protocol.json').read_text())['pins'];base=next(A.Path(f).parent for f in pins if f.endswith('error_captured.npz'))
 for f,h in pins.items():assert A.hashlib.sha256(A.Path(f).read_bytes()).hexdigest()==h
 fixture=json.loads((base/'negative_mass_fixture.json').read_text());cases.append((np.array(fixture['rays']),'fixture672',fixture['positive'],fixture['bound'],'ordinary'))
 rays=np.load(base/'error_captured.npz')['rays'];refs={r['index']:r for i in range(8) for r in json.loads((A.HERE.parent/f'full_{i}.json').read_text())['rows']}
 for idx in A.P['anomaly_indices']:
  r=refs[idx];cases.append((rays[idx],'prior_anomaly_'+str(idx),r['reference'],r['reference_bound'],'actual_anomaly'))
 for k in [10,20,24,30,40]:
  eps=2.**(-k);a=-math.sqrt(1-eps*eps);R=np.eye(4);R[1]=[a,eps,0,0];p=math.atan2(eps,a)/(8*math.pi);cases.append((R,'antipodal_'+str(k),p,0,'general_domain_stress'))
 for R,label,p,b,kind in cases:
  r,a=runner.case(R,label,p,b);r['kind']=kind;rows.append(r)
  for key,v in a.items():arrays[label+'_'+key]=v
 out=dict(rows=rows,ordinary_gate_pass=all(r['parity'] and abs(r['difference'])<=1e-10+r['reference_bound'] for r in rows if r['kind']!='general_domain_stress'),stress_gate_pass=all(r['parity'] and abs(r['difference'])<=1e-10 for r in rows if r['kind']=='general_domain_stress'),cpu=time.process_time()-START,rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 np.savez_compressed(A.HERE/'control_nodes.npz',**arrays);(A.HERE/'controls.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}));print(json.dumps([{k:r[k] for k in ['label','value','reference','difference']} for r in rows if r['kind']=='general_domain_stress']))
if __name__=='__main__':main()
