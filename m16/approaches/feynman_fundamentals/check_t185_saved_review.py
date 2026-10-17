"""Independent saved-decimal audit; no estimator imports or random draws."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal
import hashlib,json,math
from scipy.stats import binom
from scipy.optimize import brentq
HERE=Path(__file__).resolve().parent
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/t185_evidence_review')
def read(p):return json.loads(p.read_text(),parse_float=Decimal)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=read(P/'results.json'); manifest=read(P/'manifest.json')
for r in manifest['files']:
    p=P/'snapshot'/r['path'];assert len(p.read_bytes())==r['bytes'] and sha(p)==r['sha256']
B=P/'snapshot/m16/approaches/estT_fix'
ex=read(B/'run5/zoo0_exact.json');re=read(B/'run5/zoo0_recipe108.json')
p=F(ex['acc']);z=F(re['acc']);cost=F(2500,3);n=F(re['ops'])/cost
def ratio(q,draws=n):return draws*(z-q)**2/(q*(1-q))
def eq(field,value):assert F(field['fraction'])==value and float(field['float'])==float(value)
o=a['one_model'];eq(o['estimate'],z);eq(o['numerical_reference'],p)
eq(o['difference'],z-p);eq(o['difference_percentage_points'],100*(z-p))
eq(o['legacy_sample_cost'],cost);eq(o['legacy_fractional_draws'],n)
assert o['floor_draws']==n.numerator//n.denominator
eq(o['nominal_ratio'],ratio(p));eq(o['floor_ratio'],ratio(p,int(n)))
eq(o['recipe_to_exact_proxy_cost'],F(re['ops'],ex['ops']))
eq(o['recipe_proxy_per_cell'],F(re['ops'],re['cells']))
eq(o['timing_ratio_with_leaf_diagnostic'],F(re['sec'])/F(ex['sec']))
mc=F(read(B/'mc_H8_T5.json')['zoo0']);N=10_000_000;k=int(mc*N)
assert mc*N==k==a['mc']['reconstructed_hits_not_saved_ledger']
eq(a['mc']['raw_mc_point_ratio'],ratio(mc))
# Invert binomial tails directly, rather than the author's beta inverse.
lo=brentq(lambda q:binom.sf(k-1,N,q)-.025,float(mc)-.01,float(mc),xtol=5e-15)
hi=brentq(lambda q:binom.cdf(k,N,q)-.025,float(mc),float(mc)+.01,xtol=5e-15)
stored=list(map(float,a['mc']['cp95_numeric']))
assert max(abs(x-y) for x,y in zip(stored,[lo,hi]))<1e-12
ends=[ratio(F.from_float(x)) for x in stored]
assert stored[0]<float(z)<stored[1]
assert a['mc']['ratio_interval'][0]==0
assert math.isclose(float(max(ends)),float(a['mc']['ratio_interval'][1]),rel_tol=1e-11)
for x in o['acceptable_true_probability_interval']:
    assert abs(float(ratio(F(x)))-1)<1e-8
v=read(B/'val5.json');rows=[[F(x) for x in r] for r in v['thin_rows(det,pl256,d128,d64,th4-10,th6-10,th10-pl)']]
assert len(rows)==60 and all(len(r)==7 for r in rows)
cols=dict(log128=2,log64=3,thin4=4,thin6=5,thin10_vs_plackett=6)
av=a['validation']
for name,c in cols.items():
    assert float(max(abs(r[c]) for r in rows))==float(av['max_abs_columns'][name])
    assert math.isclose(float(max(abs(r[c]/r[1]) for r in rows)),float(av['max_relative_columns'][name]),rel_tol=1e-15)
series={'pl256':[r[1] for r in rows],'pl128':[r[1]+r[2] for r in rows],
        'pl64':[r[1]+r[3] for r in rows],'thin10':[r[1]+r[6] for r in rows],
        'thin6':[r[1]+r[6]+r[5] for r in rows]}
for name,vals in series.items():
    assert float(min(vals))==float(av['derived_series_minima'][name])
    assert sum(x<0 for x in vals)==av['derived_series_negative_counts'][name]
thin=[r for r in rows if r[0]<F(1,1_000_000)]
assert len(thin)==av['actual_thin_branch_rows']==57
assert float(max(abs(r[5]+r[6]) for r in thin))==float(av['max_actual_thin_vs_pl256_abs'])
assert v['halves_sum_minus_1']==av['halves_sum_minus_1']>0
t4=read(B/'rescore/zoo0.json')['recipe108'];g=a['growth']
eq(g['same_zoo0_T4_to_T5_cells'],F(re['cells'],t4['cells']))
eq(g['same_zoo0_T4_to_T5_proxy_ops'],F(re['ops'],t4['ops']))
eq(g['assumed_13_5_growth_5_steps'],F(27,2)**5)
eq(g['assumption_based_T10_proxy_cost'],re['ops']*F(27,2)**5)
out=dict(complete=True,manifest_files=len(manifest['files']),validation_rows=len(rows),thin_rows=len(thin),
         nominal_ratio=float(ratio(p)),mc_ratio_upper=float(max(ends)),independent_binomial_interval=[lo,hi],
         saved_interval_difference=max(abs(x-y) for x,y in zip(stored,[lo,hi])),
         negative_log64_values=sum(x<0 for x in series['pl64']),source_sha256=sha(B/'estT.py'),
         results_sha256=sha(P/'results.json'),new_model_mass_or_random_calls=0,
         scope='One saved clean case, incomplete abstract cost and source-documented MC provenance. Numerical interval inversion is not outward rounding. No certified win/failure or T10 growth theorem.')
(HERE/'t185_saved_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
