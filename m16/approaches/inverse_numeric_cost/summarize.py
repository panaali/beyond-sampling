from pathlib import Path
import hashlib,json,time
P=Path(__file__).resolve().parent
OLD=P.parent/'inverse4_cost_review'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 t=time.process_time();controls=json.loads((P/'controls.json').read_text());old=json.loads((OLD/'summary.json').read_text())
 assert controls['complete'] and controls['baseline_parity'] and controls['dispatch_restored']
 assert [(r['n'],r['cases'],r['max_abs_lanes'],r['max_cmp_lanes']) for r in controls['iamax_cases']]==[(1,8,2,7),(2,64,8,11),(3,512,8,11),(4,4096,14,15)]
 assert len(controls['inverse_controls'])==8
 cats=dict(iamax_abs_lanes=32,iamax_max_and_comparison_lanes=44,lu_pivot_abs_lanes=8,lu_pivot_comparisons=8,scaling_zero_comparisons=6,triangular_beta_comparisons=4)
 added=sum(cats.values());assert added==102
 total=old['per_inverse_envelope']+added;assert total==290
 parents=[]
 for r in old['parent_compositions']:
  count=r['inverse_calls'];parents.append(dict(label=r['label'],parent_sha256=r['parent_sha256'],calls=count,floating_expression_upper=188*count,numeric_guard_upper=added*count,combined_numeric_upper=total*count,visible_plus_inverse_numeric_subset=r['parent_visible_explicit']+total*count,complete_cost=False))
 pins={f:sha(OLD/f) for f in ['environment.json','runtime_dispatch.json','kernel_spy.c','idamax_k_SKYLAKEX_disassembly.txt','dgetf2_k_disassembly.txt','dscal_k_SKYLAKEX_disassembly.txt','dtrsm_LNLU_disassembly.txt','dtrsm_LNUN_disassembly.txt','summary.json']}
 out=dict(added_numeric_categories=cats,added_numeric_units=added,combined_numeric_units=total,parent_compositions=parents,source_pins=pins,controls_sha256=sha(P/'controls.json'),cpu=time.process_time()-t,complete_category_audit=True,complete_cost=False,scope='Conservative FP/absolute-value/comparison/max lanes only. Includes inactive packed lanes; excludes bitmask setup/shuffles, integer/index/control, copying/allocation/interpreter and IEEE side effects. Not a matched-budget proof.')
 (P/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(total=total,parents=parents,cpu=out['cpu'])))
if __name__=='__main__':main()
