"""Exact hit/state contraction and fixed-size binomial interval; no MSE flooring."""
import hashlib,json,math,zlib
from pathlib import Path
from scipy.stats import beta
HERE=Path(__file__).resolve().parent
identity=json.loads((HERE/'noise_identity.json').read_text());pins=json.loads((HERE/'reference_pins.json').read_text())
rows=[json.loads((HERE/'noise_reference'/f'{i:03d}.json').read_text()) for i in range(100)]
for i,r in enumerate(rows):
 assert r['complete'] and r['index']==i and r['n']==1_000_000 and r['planned_total_n']==100_000_000
 assert r['model_sha256']==identity['model_sha256'] and r['source_pins']==pins and r['seed_crc32']==zlib.crc32(identity['name'].encode())
 assert len(r['batches'])==2 and all(b['n']==500_000 for b in r['batches']) and sum(b['hits'] for b in r['batches'])==r['hits']
 if i:assert r['rng_before']==rows[i-1]['rng_after']
hits=sum(r['hits'] for r in rows);n=sum(r['n'] for r in rows);p=hits/n
ci=[float(beta.ppf(.025,hits,n-hits+1)),float(beta.ppf(.975,hits+1,n-hits))]
out={'model_sha256':identity['model_sha256'],'hits':hits,'n':n,'accuracy':p,'standard_error':math.sqrt(p*(1-p)/n),'binomial_CP95':ci,'reference_cpu_seconds':sum(r['cpu_seconds'] for r in rows),'max_rss_kib':max(r['max_rss_kib'] for r in rows),'extra_parity_evaluations':sum(r.get('extra_parity_evaluations',0) for r in rows),'continuous_rng_chain_verified':True,'scope':'Original fixed100M Monte Carlo stream; statistical uncertainty under iid reference model, float64 network semantics. No exact-real rounding certificate or multi-model rung claim.'}
(HERE/'reference_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
