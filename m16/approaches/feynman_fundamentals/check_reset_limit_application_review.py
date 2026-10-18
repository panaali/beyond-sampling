"""Trained saved-weight sector audit, no original network evaluation."""
from check_reset_limit_review import *
a=read(P/'application.json');parent=read(P.parent/'application.json');gate=read(P/'application_gate.json')
assert a['complete'] and a['dependency_gate']==gate
for name,h in gate['source_sha256'].items():assert sha(P/name)==h
assert a['application_source_sha256']==sha(P/'application.py') and a['weights']==parent['weights']
assert a['checkpoint_loads']==a['original_network_forward_MC_calls']==0
wi=list(map(F,a['weights']['Wi']));wh=[list(map(F,row)) for row in a['weights']['Wh']]
assert list(map(F,a['partition']['A']))==[wh[i][1]*wi[1] for i in range(2)]
assert list(map(F,a['partition']['B']))==wi and a['partition']['Wo']==a['weights']['Wo']
value=check(a['partition'],a['mass'])
lo,hi=endpoints(a['mass']['joint_correctness_mass'])
event=F(parent['bounds']['event_probability']['upper']);sector=F(1,12)
assert F(a['inherited_unconditional_disagreement_upper'])==event
expected=(max(F(0),lo-event),min(sector,hi+event))
assert endpoints(a['original_joint_correctness_enclosure'])==expected
assert endpoints(a['original_conditional_sector_accuracy_enclosure'])==tuple(x/sector for x in expected)
assert endpoints(a['limit_conditional_sector_accuracy_enclosure'])==(lo/sector,hi/sector)
assert a['exact_reduced_formula_representative_evaluations']==sum(len(r['representatives']) for r in a['partition']['intervals'])==14
assert a['exact_reduced_formula_boundary_evaluations']==len(a['partition']['boundaries'])==6
c=F(6341485325517743533,170399438932673560576);assert 0<c<1
for cell in a['partition']['intervals']:
    for rep in cell['representatives']:assert rep['winner']==(2 if F(rep['r'])>c else 1)
boundary=next(r for r in a['partition']['boundaries'] if F(r['r'])==c);assert boundary['winner']==1 and boundary['maximizers']==[1,2]
assert abs(value-(m(sector)-cdf('10',c,False)/2))<mp.mpf('1e-170')
out=dict(complete=True,intervals=7,boundaries=6,reduced_representatives=14,crossing=str(c),
         joint_limit_display=[float(lo),float(hi)],original_joint_display=list(map(float,expected)),
         original_conditional_sector_display=[float(x/sector) for x in expected],
         application_sha256=sha(P/'application.json'),new_original_network_calls=0,
         scope='Exact-real stored dyadic H2T4 weights and one negative-prefix/positive-target second-rank sector. Saved complete partition, first-index crossing, independent arcsine CDF, inherited event upper and normalization checked. Inherited exhaustive face proof remains a premise; no full-model accuracy, floating-forward or production-cost claim.')
(HERE/'reset_limit_application_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
