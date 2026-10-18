"""Frozen saved-data-only discriminator; refuses overwrites, preserves failures."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,hashlib,time,resource,traceback
import mpmath as mp
from primitive import Bound,certificate
P=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check_formula(cert,row):
 with mp.workdps(120):
  def m(x):v=F(x);return mp.mpf(v.numerator)/v.denominator
  a,b=m(row['a_expanded']),m(row['b_expanded']);c=(a+b)/2;r=(b-a)/(b+a);N=row['degree'];coef=mp.mpf(1);s=mp.mpf(1)
  for k in range(1,N+1):coef*=mp.mpf(2*k+3)/(2*k);s+=coef*r**k
  tail=mp.mpf(0) if a==b else a**(-mp.mpf('2.5'))-c**(-mp.mpf('2.5'))*s
  v=abs(m(cert['det']))*tail/(32*mp.pi**2)
  assert m(row['lower_formula_endpoint'])<=v<=m(row['upper_bound'])
  return mp.nstr(v,120)
def main():
 out=P/'results.json';assert not out.exists(),'immutable attempt already exists'
 protocol=json.loads((P/'protocol.json').read_text());assert sha(protocol['input_results'])==protocol['input_sha256'];saved=json.loads(Path(protocol['input_results']).read_text());assert saved['complete'] and len(saved['fixtures'])==11
 t=time.process_time();B=Bound();d=dict(complete=False,protocol=protocol,source_sha256={p.name:sha(p) for p in [P/'primitive.py',P/'run.py',P/'protocol.json']},cold_counts=B.cold_counts,controls=[],certificates=[],rows=[])
 def save():
  d['cpu']=time.process_time()-t;d['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;out.write_text(json.dumps(d,indent=2)+'\n')
 save()
 try:
  for i,cert in enumerate(protocol['additional_controls']):
   for n in protocol['orders']:
    row=B.one(cert,n);row['control']=i;row['formula120']=check_formula(cert,row);d['controls'].append(row);save()
  for fixture in saved['fixtures']:
   cert=certificate(fixture['rays_hex']);cert['fixture']=fixture['name'];assert F(cert['det'])==F(fixture['exact_stored_determinant']);d['certificates'].append(cert);save()
   for n in protocol['orders']:
    row=B.one(cert,n);row['fixture']=fixture['name'];row['formula120']=check_formula(cert,row)
    prior=next(x for x in saved['rows'] if x['fixture']==fixture['name'] and x['arm']==f'duffy{n}')
    row['source_value_hex']=prior['returned_hex'];row['source_absolute_error']=prior['absolute_error'];row['observed_error_within_ideal_bound']=F(prior['absolute_error'])<=F(row['upper_bound']);d['rows'].append(row);save()
  d['complete']=True;save()
 except Exception:
  d['failure']=traceback.format_exc();save();raise
 counts=Counter();setup=Counter()
 for row in d['rows']:counts.update(row['counts'])
 for cert in d['certificates']:setup.update(cert['counts'])
 summary=dict(complete=True,rows=len(d['rows']),qualified=sum(r['qualifies'] for r in d['rows']),orders={str(n):dict(qualified=sum(r['qualifies'] for r in d['rows'] if r['order']==n),total=11) for n in protocol['orders']},exact_counts=dict(setup),directed_counts=dict(counts),cold_counts=d['cold_counts'],max_exact_bits=max(c['max_bits'] for c in d['certificates']),certificates_cpu=sum(c['cpu'] for c in d['certificates']),cpu=d['cpu'],results_sha256=sha(out),scope='Ideal GL truncation only. Saved source errors are diagnostic; source floating nodes, determinants and evaluation remain excluded. No mass rerun. Exact operation counts exclude rational internals, comparisons, allocation, conversions, and Python/control work; directed counts exclude analogous runtime work. Controls and MP validation are separate research work.')
 (P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
