"""Identified val-comparison bypass envelope, not a complete native cost bound."""
import json, math
from fractions import Fraction as F
from pathlib import Path
import summarize_direct as S
HERE=Path(__file__).resolve().parent

def one(rows,H,sampler):
 factor=1+max(F(1,3),F(4*H,3*H+8))
 enriched=[dict(r,ops=math.ceil(factor*r['ops'])+1) for r in rows]
 return dict(factor=str(factor),raw=S.summarize(rows,sampler),identified_gap_envelope=S.summarize(enriched,sampler),numeric_only_sampler_envelope=S.summarize(enriched,sampler-3))

def main():
 out=dict(scope='Conservative fill of identified _step2.margin val comparisons and final subtraction only. Other plain-zero arithmetic, integer/control/memory, model-free setup and native/library costs remain excluded. Original sampler includes three model-free rank comparisons; numeric-only alternative removes those three and excludes final integer correctness in both methods.',review='23c5c972',groups=[])
 for H in (4,5,6,8):
  for z in range(5):
   rows=[]
   for noise in (80,81,82,83):
    d=json.loads((HERE/'direct_counts'/f'core_H{H}_z{z}_n{noise}.json').read_text());assert d['complete'];rows+=d['rows']
   g=one(rows,H,d['sampler_ops']);g.update(H=H,zoo=z);out['groups'].append(g)
 d=json.loads((HERE/'direct_counts'/'borderline.json').read_text());out['borderline']=one(d['rows'],8,d['sampler_ops'])
 out['rungs']=[dict(H=H,worst_ratio=max(g['numeric_only_sampler_envelope']['ratio_integer_budget'] for g in out['groups'] if g['H']==H),largest_ci_upper=max(g['numeric_only_sampler_envelope']['bootstrap95_integer'][1] for g in out['groups'] if g['H']==H)) for H in (4,5,6,8)]
 (HERE/'identified_cost_envelope.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['rungs']))
if __name__=='__main__':main()
