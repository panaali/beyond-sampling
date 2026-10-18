"""Saved-only complete contraction; no candidate, rule or geometry evaluations."""
from collections import Counter
import json,math,time
from full_run import P,D,sha,write,load_protocol,read_rows

def main():
    start=time.process_time();proto,target,tol,pilot,saved=load_protocol();rows,sessions=read_rows(proto,tol,saved)
    assert len(rows)==proto['N'] and sum(r['reused_pilot'] for r in rows)==64
    known=Counter();methods=Counter();orders=Counter();unresolved=[];rescues=[]
    for row in rows:
        c=row['candidate'];methods[c.get('method',c['mode'])]+=1
        known.update(c['center']['setup_selection_counts']);known.update(c.get('repeated_determinant_counts',{}));known.update(c.get('kernel_counts',{}))
        # M.one omits top-level kernel counts if a rescue evaluates then misses its series target.
        if 'kernel_counts' not in c and 'kernel_counts' in c.get('rescue',{}):known.update(c['rescue']['kernel_counts'])
        if c['mode']!='returned':unresolved.append(row['index'])
        if 'rescue' in c:rescues.append(dict(index=row['index'],exact_inventory=c['rescue']['exact_operation_inventory'],stages=[{k:s[k] for k in ('method','certificate_cpu','selection_cpu') if k in s} for s in c['rescue']['stages']]))
        if c.get('order') is not None:orders[c['order']]+=1
    events=pilot['candidate_rule_events']+[e for s in sessions for e in s['rule_events']]
    canonical={};staged=Counter()
    for e in events:
        staged.update(e['counts']);n=e['order']
        if n in canonical:assert canonical[n]['counts']==e['counts']
        else:canonical[n]=e
    for e in canonical.values():known.update(e['counts'])
    known['universal_constant_arithmetic']+=6
    good=[r['candidate'] for r in rows if r['candidate']['mode']=='returned']
    probability=math.fsum(c['value'] for c in good);bound=math.fsum(c['bound'] for c in good)
    out=dict(complete=not unresolved,N=len(rows),protocol_sha256=sha(P/'full_protocol.json'),pilot_reused=64,
       shard_hashes={p.name:sha(p) for p in sorted(D.glob('shard_*.jsonl'))},session_hashes={p.name:sha(p) for p in sorted(D.glob('session_*.json'))},
       returned=len(good),unresolved_indices=unresolved,methods=dict(methods),orders=dict(orders),
       covered_error_probability=probability,conditional_bound_sum=bound,known_float_counts=dict(known),known_float_total=sum(known.values()),
       canonical_rule_events=list(canonical.values()),staged_rule_counts=dict(staged),staged_rule_total=sum(staged.values()),
       exact_rescues=rescues,exact_and_mp_setup_in_known_float_total=False,full_cost=False,full_floating_certificate=False,
       cpu_sessions=sum(s['cpu'] for s in sessions),pilot_total_cpu_separate=pilot['cpu'],maxrss_kib=max(s['maxrss_kib'] for s in sessions),
       candidate_cpu_new=sum(r['candidate']['cpu'] for r in rows if not r['reused_pilot']),
       adapter_cold_setup_and_validation_separate=True,geometry_legacy_proxy_separate=38450554,
       no_matched_cost_or_rung_claim=True,summary_cpu=time.process_time()-start)
    if not unresolved:
        out.update(accuracy=1-probability,published_exact_accuracy=proto['published_exact_accuracy'],
                   delta_from_published_exact=1-probability-proto['published_exact_accuracy'],published_comparison_is_diagnostic=True)
    write(P/'full_summary.json',out)
    print(json.dumps({k:out[k] for k in ('complete','N','returned','unresolved_indices','known_float_total','conditional_bound_sum','cpu_sessions')},allow_nan=False))
if __name__=='__main__':main()
