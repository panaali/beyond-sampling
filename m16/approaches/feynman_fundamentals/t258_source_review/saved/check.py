"""Independent retained-summary audit. Stdlib only, never imports producer code."""
import hashlib,json,math,os,pathlib,resource,time
from fractions import Fraction as Q
P=pathlib.Path(__file__).resolve().parent
B=pathlib.Path('/workspace/work/recover-small_h2_56/review/replacement_claims_recheck')
START=time.process_time(); OUT=P/'output'; R={'complete':False,'pass_all':False,'counts':{},'checks':{}}
def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def save():
 t=OUT/'result.tmp'
 with t.open('w') as f:json.dump(R,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,OUT/'results.json')
def event(k):
 with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps({'kind':k})+'\n');f.flush();os.fsync(f.fileno())
 assert time.process_time()-START<17
 save()
def pins(d):
 for p,h in d.items():assert digest(p)==h,p
def check(name,ok,**data):
 assert name not in R['checks'];R['checks'][name]={'name':name,'passed':bool(ok),**data}
 if not ok:save();raise AssertionError(name)
def count(tag,p,n):
 assert tag not in R['counts'] and type(n)is int and 0<n<=10**6
 assert type(p)in(int,float) and math.isfinite(p) and 0<=p<=1
 x=Q(float(p))*n;floor=x.numerator//x.denominator
 # Under binary64 rounding, n*|p-k/n| < 10^-9: only adjacent integers can qualify.
 candidates=[k for k in range(max(0,floor-1),min(n,floor+2)+1) if float(Q(k,n))==float(p)]
 R['counts'][tag]={'denominator':n,'float_hex':float(p).hex(),'compatible_counts':candidates}
 assert len(candidates)==1,(tag,candidates)
 return candidates[0]
def rate(tag,k,n,p):
 if n==0:check(tag,p is None,expected_none=True)
 else:check(tag,float(Q(k,n))==p,numerator=k,denominator=n,saved=p)
def metrics(tag,row,n):
 for k in ('acc','agree'):
  if k in row:count(tag+'/'+k,row[k],n)
def main():
 scope=read(P/'scope.json');pins(scope['pins']);g=read(P/'gate.json');pins(g['pins'])
 review=read(g['review_path']);assert review['complete'] and review['pass_all'] and review['execution_allowed']
 for p in (str(P/'check.py'),str(P/'scope.json'),str(P/'PROTOCOL.md')):
  assert p in g['pins'] and p in review['pins'] and g['pins'][p]==review['pins'][p]
 assert g['review_path'] in g['pins'];pins(review['pins'])
 for k in ('review_commit','publication'):
  assert 8<=len(g[k])<=40 and all(c in '0123456789abcdef' for c in g[k])
 cpu=resource.getrlimit(resource.RLIMIT_CPU);mem=resource.getrlimit(resource.RLIMIT_AS)
 R['runtime']={'cpu':cpu,'memory':mem};assert cpu==(20,20) and mem==(2**31,2**31)
 producer_gate=read(B/'saved/gate.json');pins(producer_gate['pins']);pins(read(B/'saved/protocol.json')['pins'])
 for key in ('source_review_path','written_review_path'):
  r=read(producer_gate[key]);assert r['complete'] and r['pass_all'];pins(r['pins'])
 inv=read(B/'retained/inventory.json');assert len(inv['records'])==7
 for r in inv['records']:assert r['present'] and digest(B/'retained'/r['snapshot'])==r['sha256']
 event('begin')
 C=read(B/'retained/results/phaseC2.json');V=read(B/'retained/results/phaseC.json');D=read(B/'retained/results/phaseD.json');E=read(B/'retained/results/phaseC_diag.json')
 n=D['N'];check('sample_sizes',n==200000 and C['N']==V['N']==[1000000,200000,200000])
 names=C['families'];check('families',len(names)==12 and len(set(names))==12)
 for f in ('val_r2','eval_r2'):
  a=C[f];check(f+'/shape',len(a)==12 and all(len(t)==11 and all(len(v)==16 and all(math.isfinite(z) for z in v) for v in t) for t in a))
 selected=[]
 for t in range(1,11):
  row=[]
  for j in range(16):
   best=0
   for f in range(1,12):
    if C['val_r2'][f][t][j]>C['val_r2'][best][t][j]:best=f
   row.append(names[best])
  selected.append(row)
 check('validation_selection',selected==C['best_family'],reconstructed=selected)
 for choice in ('best','best_noH'):
  group=C['cumulative'][choice];last=C['cumulative'][choice+'_final_step_only']
  check(choice+'/groups',len(group)==6 and group[-1]['S']==[4,2,5,9,1,6,7,8,3,13,0,10,11,12,14,15])
  check(choice+'/final_identity',all(group[-1][k]==last[k] for k in last))
  for i,row in enumerate(group):metrics(choice+'/cumulative/'+str(i),row,n)
  for kind in ('single','group'):
   for label,row in C[kind][choice].items():metrics(choice+'/'+kind+'/'+label,row,n)
 order=sorted(range(16),key=lambda j:(-C['single']['best']['n'+str(j)]['agree'],j))
 check('evaluation_selected_order',[x['added'] for x in C['cumulative_by_neuron']]==['n'+str(j) for j in order],order=order)
 check('per_neuron_endpoint',all(C['cumulative_by_neuron'][-1][k]==v for k,v in C['cumulative']['best_final_step_only'].items()))
 event('selection')
 keys=['net_ok&rep_ok','net_ok&rep_wrong','net_wrong&rep_ok','net_wrong&rep_wrong_same','net_wrong&rep_wrong_diff']
 conf={k:count('confusion/'+k,p,n) for k,p in D['confusion'].items()};check('confusion_keys',list(conf)==keys)
 a,b,c,d,e=[conf[k] for k in keys];dis=b+c+e;net=a+b;rep=a+c
 check('confusion_partition',a+b+c+d+e==n)
 for tag,k,den,p in [('disagreement',dis,n,D['disagree']),('C2_base',net,n,C['base']['acc']),('C2_accuracy',rep,n,C['cumulative']['best'][-1]['acc']),('C2_agreement',n-dis,n,C['cumulative']['best'][-1]['agree']),('error_coverage',c+e,n-net,D['disagree_given']['net_wrong']),('correct_disagreement',b,net,D['disagree_given']['net_correct'])]:rate(tag,k,den,p)
 paired={'gain_count':c-b,'gain_exact':str(Q(c-b,n)),'population_variance_exact':str(Q(n*(b+c)-(c-b)**2,n*n)),'mean_sample_variance_exact':str(Q(n*(b+c)-(c-b)**2,n*n*(n-1)))}
 R['paired']=paired
 cases={}
 for name,row in D['by_case'].items():
  support=count('case/'+name+'/support',row['share'],n);hits=count('case/'+name+'/hits',row['disagree'],support)
  cases[name]=(support,hits);rate('case/'+name+'/share_dis',hits,dis,row['share_of_disagreements'])
 check('exclusive_cases',tuple(map(sum,zip(*(cases[k] for k in ('x9max','x9second','x9below')))))==(n,dis))
 near=D['near_tie_or_blind'];ns=count('near/support',near['share'],n);nh=count('near/hits',near['disagree'],ns)
 rate('near/share_dis',nh,dis,near['share_of_disagreements']);rate('far/dis',dis-nh,n-ns,near['disagree_elsewhere'])
 totals={}
 for name in ('by_gap23','by_gap12','by_full_gate_pattern_count','by_full_gate_pattern_count_near','by_full_gate_pattern_count_far'):
  rows=[]
  for i,row in enumerate(D[name]):
   tag=name+'/'+str(i);s=count(tag+'/support',row['share'],n)
   if s:h=count(tag+'/hits',row['disagree'],s);err=count(tag+'/errors',row['net_err'],s)
   else:check(tag+'/empty',row['disagree'] is None and row['net_err'] is None);h=err=0
   if 'share_of_disagreements'in row:rate(tag+'/share_dis',h,dis,row['share_of_disagreements'])
   rows.append([s,h,err])
  total=[sum(x[j] for x in rows) for j in range(3)];totals[name]=total
  expected=(ns,nh) if name.endswith('_near') else (n-ns,dis-nh) if name.endswith('_far') else (n,dis)
  check(name+'/partition',total[:2]==list(expected),counts=rows,totals=total)
  if not name.endswith(('_near','_far')):check(name+'/errors',total[2]==n-net)
 check('near_far_errors',totals['by_full_gate_pattern_count_near'][2]+totals['by_full_gate_pattern_count_far'][2]==n-net)
 fe=totals['by_full_gate_pattern_count_far'][2];far=D['far_disagree_given'];rate('far/error_rate',fe,n-ns,far['net_err_far'])
 bad=count('far/error_disagree',far['net_wrong'],fe);good=count('far/correct_disagree',far['net_correct'],n-ns-fe);check('far/disjoint_errors',bad+good==dis-nh)
 check('rank_histogram_keys',set(D['rank_picked_when_disagree'])=={'replacement','network'})
 for key,ar in D['rank_picked_when_disagree'].items():check('rank/'+key,len(ar)==10 and all(type(z)is int and z>=0 for z in ar) and sum(ar)==dis,counts=ar)
 check('noise_rows',len(E['noise'])==7 and [r['1-R2'] for r in E['noise']]==[.01,.003,.001,.0003,.0001,.00003,.00001])
 for i,row in enumerate(E['noise']):metrics('noise/'+str(i),row,n)
 for label,row in E.items():
  if label!='noise':
   for key,p in row.items():
    if key.endswith(('_agree','_acc')):count('direct/'+label+'/'+key,p,n)
 log=(B/'retained/logs/phaseC2.log').read_text().splitlines();check('fit_log_steps',[s for s in log if s.startswith('fit t ')]==['fit t '+str(t) for t in range(1,11)])
 event('reconstructed')
 got=read(B/'saved/output/result.json');assert got['complete'] and got['pass_all'] and not got['unknowns']
 assert len(got['records'])==len(R['checks'])==63 and len(got['counts'])==len(R['counts'])==237
 assert {r['name']:r for r in got['records']}==R['checks']
 for tag,v in R['counts'].items():
  assert all(got['counts'][tag][k]==x for k,x in v.items()),tag
  lo,hi=got['counts'][tag]['rounding_interval_integer_range'];assert lo<=v['compatible_counts'][0]<=hi and hi-lo<=1
 assert set(got['counts'])==set(R['counts']) and got['paired']==paired
 events=[json.loads(s) for s in (B/'saved/output/progress.jsonl').read_text().splitlines()];assert [r['kind'] for r in events]==['begin','selection_complete','partition_complete','complete']
 assert got['runtime']['cpu']==[20,20] and got['runtime']['memory']==[2**31,2**31]
 assert 0<=got['cpu_seconds']<17
 R['producer_cpu_seconds']=got['cpu_seconds'];R['scope']='Conditional retained-summary compatibility, no raw prediction or historical runtime certification'
 R['cpu_seconds']=time.process_time()-START;event('complete');R['complete']=R['pass_all']=True;save()
if __name__=='__main__':
 assert not OUT.exists();OUT.mkdir()
 try:main()
 except BaseException as exc:
  R['complete']=R['pass_all']=False;R['fatal']={'type':type(exc).__name__,'message':str(exc),'cpu_seconds':time.process_time()-START};save()
  with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps({'kind':'fatal',**R['fatal']})+'\n')
  raise
