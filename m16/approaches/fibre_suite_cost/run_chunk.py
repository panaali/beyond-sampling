"""Bounded native recount of unchanged existing T99 instances; no new truths."""
import time
START=time.process_time()
import argparse,json,resource,hashlib
from .common import HERE,verify_sources,models,references,digest,evaluate,sample_count

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--H',type=int,choices=(4,5,6),required=True);parser.add_argument('--start',type=int,required=True);parser.add_argument('--count',type=int,default=20);args=parser.parse_args();assert 0<=args.start<157 and 1<=args.count<=20
    resource.setrlimit(resource.RLIMIT_CPU,(110,110));assert json.loads((HERE/'preflight.json').read_text())['complete'];checks=verify_sources();panel=models(args.H);old,prior=references(args.H);names=list(panel)[args.start:args.start+args.count];dest=HERE/f'h{args.H}_{args.start:03d}.json';assert not dest.exists(),'Do not overwrite prior or partial evidence';out=dict(H=args.H,start=args.start,expected=len(names),rows=[],complete=False,source_equivalence=checks,source_manifest_sha256=hashlib.sha256((HERE/'source_manifest.json').read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),numerical_truths_interval_certified=False,complete_implementation_cost=False)
    def save():out['cpu_including_imports']=time.process_time()-START;dest.write_text(json.dumps(out,indent=2)+'\n')
    for index,name in enumerate(names,start=args.start):
        model=panel[name];p=prior[name];previous=old[name];sha=digest(model);assert sha==p['weight_sha256'];start=time.process_time();row=dict(index=index,name=name,weight_sha256=sha,truth=previous['truth'],original_estimate=previous['reference']['est'],original_counter=previous['reference']['ops']);out['rows'].append(row)
        try:
            r=evaluate(model);row['native']=r;ref=p['methods']['reference'];parity=dict(value=r['value_hex']==float(previous['reference']['est']).hex(),saved_t99_value=r['value_hex']==float(ref['estimate']).hex(),counter=r['original_n']==previous['reference']['ops']==ref['proxy_ops'],traces=r['traces']==ref['info']['traces']==previous['reference']['traces'],cons=r['cons']==ref['info']['cons']==previous['reference']['cons'],segments=r['segments']==ref['info']['segments'],families=r['family_hex']=={k:float(v).hex() for k,v in ref['error_by_family'].items()});row['parity']=parity;assert all(parity.values()),'Pinned existing value/counter/family parity mismatch'
            sample=sample_count(model);sampletotal=sample['named_numeric_total']+sample['integer_control_total'];assert sampletotal==12*args.H+4*args.H**2;N=r['expression_total']/sampletotal;ptrue=row['truth'];bias=r['value']-ptrue;oldsample=3*args.H*(args.H+2);oldN=r['original_n']/oldsample;row.update(sampler=sample,sampler_total=sampletotal,expression_sample_equivalents=N,original_sample_equivalents=oldN,equivalent_budget_multiplier=N/oldN,bias=bias,sampling_variance=ptrue*(1-ptrue)/N,squared_bias_ratio=bias*bias*N/(ptrue*(1-ptrue)),original_squared_bias_ratio=bias*bias*oldN/(ptrue*(1-ptrue)),inventory_cpu=time.process_time()-start)
        except Exception as exc:row['failure']=type(exc).__name__+': '+str(exc);out['stop_reason']='Preserved first mismatch/failure';save();print(json.dumps(dict(H=args.H,name=name,failure=row['failure'])),flush=True);return
        save()
        if (index-args.start+1)%5==0:print(json.dumps(dict(H=args.H,completed=len(out['rows']),last=name,cpu=out['cpu_including_imports'])),flush=True)
        if time.process_time()-START>90:out['stop_reason']='90CPU soft cap';save();return
    out['complete']=True;save();print(json.dumps(dict(H=args.H,start=args.start,complete=True,count=len(out['rows']),cpu=out['cpu_including_imports'],max_ratio=max(r['squared_bias_ratio'] for r in out['rows']),budget_multiplier_range=[min(r['equivalent_budget_multiplier'] for r in out['rows']),max(r['equivalent_budget_multiplier'] for r in out['rows'])])),flush=True)
if __name__=='__main__':main()
