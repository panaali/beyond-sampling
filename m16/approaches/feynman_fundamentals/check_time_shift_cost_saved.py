from pathlib import Path
import hashlib,json,resource,time
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
P=Path('/workspace/work/recover-small_h2_910/review/time_shift_reuse/census');O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def audit(r):
 H,T=r['H'],r['T'];words=r['unique_histories'];assert words==sorted(set(words)) and all(len(w)==H*T and set(w)<={'0','1'} for w in words)
 pref=[sorted({w[:H*t] for w in words}) for t in range(T+1)];assert pref==r['parent_prefixes']
 work=lambda n:{'mul':H*H*n,'add':H*(H-1)*n};plus=lambda x,y:{k:x[k]+y[k] for k in x}
 baseline=work(sum(t*len(pref[t]) for t in range(T)));assert baseline==r['baseline']
 assert r['baseline_sites']==[work(t*len(pref[t])) for t in range(T)] and r['input_rows']-len(words)==r['duplicate_rows']
 assert [c['cut'] for c in r['cuts']]==list(range(1,T))
 for c in r['cuts']:
  a=c['cut'];b=T-a;suf=[sorted({w[H*a:H*(a+j)] for w in words}) for j in range(b)];assert suf==c['suffix_parent_prefixes']
  cold=work(sum((H+j)*len(suf[j]) for j in range(1,b)));prefix=work(sum(t*len(pref[t]) for t in range(a)));subs=work(a*sum(len(pref[t]) for t in range(a,T)));warm=plus(prefix,subs);combined=plus(warm,cold)
  for key,val in [('cold',cold),('prefix_work',prefix),('substitutions',subs),('warm',warm),('combined',combined)]:assert c[key]==val
  assert c['substitution_sites']==[work(a*len(pref[t])) for t in range(a,T)]
  assert c['cold_sites']==[work(0 if j==0 else (H+j)*len(suf[j])) for j in range(b)]
  assert c['named_saving']==sum(baseline.values())-sum(combined.values())
  assert c['logical_cache_preactivation_entries']==sum(len(suf[j])*H*(H+j+1) for j in range(b))
def main():
 p=json.loads((P/'protocol.json').read_text())
 for file,h in p['pins'].items():assert sha(Path(file))==h,file
 r=json.loads((P/'controls.json').read_text());assert r['complete'] and r['protocol_sha256']==sha(P/'protocol.json')
 names=['single','duplicates','last_branch','first_branch','all_H1T3','all_H1T4','all_H2T3','shared_suffix_H2T4'];assert [x['name'] for x in r['rows']]==names
 for row in r['rows']:
  audit(row['result']);assert sum(row['result']['baseline'].values())==row['expected_baseline'];assert [sum(c['combined'].values()) for c in row['result']['cuts']]==row['expected_combined']
 assert [x['name'] for x in r['invalid']]==['empty','ragged','nonbinary','invalidH'] and all(x['status']=='rejected' for x in r['invalid'])
 result=dict(complete=True,valid=8,invalid=4,controls_sha256=sha(P/'controls.json'),review_cpu=time.process_time()-start,scope='Independent prefix/suffix sets and closed-form multiplication/addition vectors, all cuts/cache-entry counts. No author census virtual-loop replay or model/matrix/geometry calls. Counts are projected dense named arithmetic only.')
 (O/'time_shift_cost_controls_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
