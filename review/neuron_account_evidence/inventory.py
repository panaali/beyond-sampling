"""Read only pinned JSON schema and text metadata, never binary archives."""
from pathlib import Path
import json,hashlib,subprocess,ast,time
B=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
t=time.process_time();m=json.loads((B/'manifest.json').read_text())
for r in m['files']:
 if r['copied']:
  b=(B/r['snapshot']).read_bytes();assert sha(b)==r['sha256'] and len(b)==r['size']
  assert b==subprocess.check_output(['git','show',m['author_revision']+':'+r['path']])
  if r['path'].endswith('.py'):ast.parse(b)
A=json.loads((B/'snapshot/results/phaseA.json').read_text());M=json.loads((B/'snapshot/results/phaseA_modes.json').read_text())
keys=['acc','agree','acc_x9max','acc_x9second','acc_x9below','acc_blind'];arms=['zero','mean','resample'];groups=arms+[x+'_final_only' for x in arms]
assert set(A)=={'N','seed_eval','seed_pilot','base','case_shares','matrices','groups'}
assert set(A['base'])==set(keys) and set(A['case_shares'])=={'x9max','x9second','x9below','blind'}
assert set(A['matrices'])==set(arms) and len(A['groups'])==22
for arm in arms:
 assert set(A['matrices'][arm])==set(keys)
 for v in A['matrices'][arm].values():assert len(v)==16 and all(len(row)==10 and all(type(x)is float for x in row) for row in v)
for g,d in A['groups'].items():
 assert set(d)==set(groups)
 for v in d.values():assert set(v)==set(keys) and all(type(x)is float for x in v.values())
assert set(M)=={'N','patterns'} and len(M['patterns'])==5
patterns=[]
for p in M['patterns']:
 assert set(p)=={'pattern','share','active','acc_in_pattern','modes'}
 for x in p['modes']:
  assert set(x)=={'abs','angle_deg','period','top_neurons','resample_h10','resample_h9','energy_share'}
  for step in ['resample_h10','resample_h9']:assert set(x[step])=={'acc','agree'}
 patterns.append(dict(pattern=p['pattern'],active_count=len(p['active']),modes=len(p['modes']),record_keys=sorted(p),mode_keys=sorted(p['modes'][0])))
prior=Path('/workspace/work/recover-small_h2_34/review/neuron_intervention_source')
for f in ['README.md','nacommon.py','phaseA.py','phaseA_modes.py','reportA.py']:assert (B/'snapshot'/f).read_bytes()==(prior/'snapshot'/f).read_bytes()
out=dict(complete=True,scope='Structural metadata only; no score arithmetic or binary payload reading',author_revision=m['author_revision'],tree_files=len(m['files']),copied_text_json_files=sum(r['copied'] for r in m['files']),unread_binary_tree_entries=[r for r in m['files'] if not r['copied']],phaseA=dict(top_keys=sorted(A),declared_N=A['N'],declared_seed_eval=A['seed_eval'],declared_seed_pilot=A['seed_pilot'],matrix_arms=arms,matrix_metrics=keys,matrix_shape=[16,10],group_names=list(A['groups']),group_arms=groups,explicit_paired_transition_fields=False,explicit_integer_case_denominators=False),modes=dict(top_keys=sorted(M),declared_N=M['N'],patterns=patterns,explicit_integer_pattern_denominators=False,projector_matrix_or_condition_fields=False,raw_row_predictions=False),prior_five_texts_identical=True,no_calls=dict(author_imports=0,binary_reads=0,binary_decodes=0,model=0,fit=0,rng=0,score_contractions=0),cpu_seconds=time.process_time()-t)
(B/'inventory.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(complete=True,texts=out['copied_text_json_files'],cpu=out['cpu_seconds'])))
