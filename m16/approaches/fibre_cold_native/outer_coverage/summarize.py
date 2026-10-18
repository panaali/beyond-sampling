"""Describe saved census coverage; no native price substitutions."""
from pathlib import Path
from collections import Counter
import json,hashlib
P=Path(__file__).resolve().parent
out=P/'summary.json';assert not out.exists()
r=json.loads((P/'results.json').read_text());assert r['complete']
missing=Counter()
for row in r['native_calls']:
 if row['input_elements'] is None:missing['input_shape_keys']+=1;missing['input_shape_calls']+=row['count']
 if row['output_elements'] is None:missing['output_shape_keys']+=1;missing['output_shape_calls']+=row['count']
c=r['categories'];matrix=c['matrix_mul']+c['matrix_add'];norm=c['norm_mul']+c['norm_add']+c['special_sqrt']
matrices=sorted([dict(site=s,named_units=sum(v for k,v in counts.items() if k.startswith('matrix_'))) for s,counts in r['native_sites'].items() if any(k.startswith('matrix_') for k in counts)],key=lambda x:-x['named_units'])
assert sum(x['named_units'] for x in matrices)==matrix
summary=dict(census_sha256=hashlib.sha256((P/'results.json').read_bytes()).hexdigest(),components=48,rescue_cells=144,rescue_prepare_included_once=True,named_visible_components=r['visible_after_diagnostic'],cold_and_outer_original=6980,prior_visible_upper=r['visible_after_diagnostic']+6980,reviewed_additional_cold=295652,unchanged_hybrid_named_composition=r['prior_hybrid_named_composition'],numeric=r['numeric_total'],integer=r['integer_total'],diagnostic=r['diagnostic_total'],matrix_expression_units=matrix,norm_expression_units=norm,top_matrix_sites=matrices,wrapped_norm_and_dot_shapes=[x for x in r['native_calls'] if x['group']=='norm_or_dot'],wrapped_reduction_summary=r['by_name']['.sum'],missing_shape_census=dict(missing),by_name=r['by_name'],by_group=r['by_group'],research_cpu=r['cpu'],max_rss_kib=r['max_rss_kib'],native_prices_added=0,scope='All native call counts reproduced from original data. by_name/by_group input_elements/output_elements sum only parseable integer-dimension shapes; missing_shape_census explicitly records omitted heterogeneous shapes. Matrix and norm named subsets are disjoint expression categories, not installed costs. Native binary-operation operand shapes/layouts, dtype/strides/runtime targets and libm bodies remain unproved. Norm length3/dot length3 and dominant clip dot3 are candidates for separately frozen applicability/dispatch controls; no source4/kernel4 transfer. No estimate/reference/cost-ratio change.')
assert summary['prior_visible_upper']==1838399658
assert summary['prior_visible_upper']+summary['reviewed_additional_cold']==summary['unchanged_hybrid_named_composition']
out.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k in ('matrix_expression_units','norm_expression_units','missing_shape_census','unchanged_hybrid_named_composition')}))
