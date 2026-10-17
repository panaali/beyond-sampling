"""Existing-reference discriminator, no new geometry or mass integration."""
import time
START=time.process_time()
import gzip,hashlib,json,resource
from collections import Counter
from fractions import Fraction as F
import probe as P

def main():
 prior=json.loads((P.HERE/'results.json').read_text());blob=P.SOURCE.read_bytes();data=json.loads(gzip.decompress(blob));unknown={(r['top'],r['second'],r['sign_x0']) for r in prior['charts'] if r['status']=='unknown'}
 decode=lambda p:F(int(p[0]),int(p[1]));wi=list(map(decode,data['model'][0]));wh=[list(map(decode,row)) for row in data['model'][1]];wo=[list(map(decode,row)) for row in data['model'][2]];D=max(v.denominator for v in wi+sum(wh,[])+sum(wo,[]));cv=lambda row:tuple(int(v*D) for v in row);model=(cv(wi),tuple(map(cv,wh)),tuple(map(cv,wo)),D)
 counts=Counter();candidates={k:[] for k in unknown};checks=0
 for i,simplex in enumerate(data['correct_simplices']):
  assert time.process_time()-START<5
  rays=tuple(tuple(int(v) for v in row) for row in simplex['integer_rays']);x=tuple(sum(r[j] for r in rays) for j in range(4));order=sorted(range(4),key=lambda j:(-x[j],j));key=(order[0],order[1],1 if x[0]>0 else -1);rows,_,_=P.chart(*key)
  assert all(P.dot(row,x)>0 for row in rows)
  assert all(P.dot(row,r)>=0 for row in rows for r in rays);checks+=16;counts[key]+=1
  if key in unknown and len(candidates[key])<3:candidates[key].append((i,x))
 new=[]
 for key,entries in sorted(candidates.items()):
  row=dict(top=key[0],second=key[1],sign_x0=key[2],correct_artifact_simplices=counts[key],attempts=[],status='unknown_artifact_absence')
  rows,_,_=P.chart(*key)
  for i,x in entries:
   kind,cert,rejected=P.witness(x,rows,key[1],model);assert kind=='correct';row['attempts'].append(dict(simplex_index=i,rejection=rejected,witness=cert))
   if cert is not None:row['status']='mixed_open';break
  new.append(row)
 out=dict(complete=True,prior_sha256=hashlib.sha256((P.HERE/'results.json').read_bytes()).hexdigest(),artifact_sha256=hashlib.sha256(blob).hexdigest(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (P.HERE/'probe.py',P.HERE/'artifact_check.py')},correct_simplex_count=len(data['correct_simplices']),vertex_rank_sign_checks=checks,chart_counts=[dict(top=a,second=b,sign_x0=s,simplices=n) for (a,b,s),n in sorted(counts.items())],unknown_chart_results=new,direct_checks=dict(P.COUNTS),cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 (P.HERE/'artifact_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['unknown_chart_results']));print(out['cpu_seconds_including_imports'])
if __name__=='__main__':main()
