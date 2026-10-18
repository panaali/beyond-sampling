"""Five frozen pairs on one stored model; no full covariance or new samples."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import sys,json,hashlib,time,resource
import numpy as np
import mpmath as mp
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from primitive import pack,mpq
from network_pairs import unpack,prepare_pair,evaluate_pair
from pair_reference import integrate

def main():
    start=time.process_time()
    out=HERE/'results.json';progress=HERE/'progress.jsonl';assert not out.exists() and not progress.exists()
    P=json.loads((HERE/'protocol.json').read_text());assert P['source_gate'] and P['analytic_saved_review']
    for path,sha in P['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
    g=json.loads(Path(P['geometry']).read_text());mean=json.loads(Path(P['means']).read_text());weights=json.loads(Path(P['weights']).read_text())
    assert g['complete'] and mean['complete'] and mean['all_checks_pass'] and mean['model']=='H16T4seed3'
    c=unpack(g['certificate']);assert c['H']==16 and len(c['sectors'])==34 and all(len(s['h3_rows'])==16 for s in c['sectors'])
    assert len(weights['rows'])==1 and weights['rows'][0]['name']=='H16T4seed3'
    assert c['wi']==list(map(F,weights['rows'][0]['wi'])) and c['wh']==[list(map(F,r)) for r in weights['rows'][0]['wh']]
    assert P['pairs']==[[0,0],[1,1],[15,15],[0,1],[0,15]]
    pairs=[tuple(v) for v in P['pairs']];result=dict(complete=False,rows=[],preflight_cpu=time.process_time()-start,import_and_prior_cpu=start,protocol_sha256=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest())
    def record(x):
        with progress.open('a') as f:f.write(json.dumps(pack(x),allow_nan=False)+'\n')
    def save():
        result.update(cpu=time.process_time()-start,process_cpu=time.process_time(),maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');tmp.replace(out)
    try:
        for i,j in pairs:
            row=dict(i=i,j=j,cells=[],complete=False);result['rows'].append(row);record(dict(stage='pair_attempt',i=i,j=j));save()
            for k,sector in enumerate(c['sectors']):
                record(dict(stage='cell_attempt',i=i,j=j,sector=k));v=sector['h3_rows'][i];w=sector['h3_rows'][j]
                t=time.process_time();certificate=prepare_pair(sector,v,w);lo=evaluate_pair(certificate,v,w,80);hi=evaluate_pair(certificate,v,w,120);ct=time.process_time()-t
                t=time.process_time();ref=integrate(sector,v,w);rt=time.process_time()-t
                cell=dict(sector=k,certificate=certificate,value80=lo,value120=hi,reference=ref,candidate_cpu=ct,reference_cpu=rt)
                if hi.get('value') is None:cell.update(passed=False,reason='unsupported retained')
                else:
                    with mp.workdps(160):
                        x=mp.mpf(hi['value']);y=mp.mpf(ref['value']);e=abs(x-y)
                        cell.update(absolute_error=mp.nstr(e,100),passed=bool(e<=mp.mpf('1e-50')+mp.mpf('1e-40')*abs(y) and abs(x-mp.mpf(lo['value']))<=mp.mpf('1e-60')*(1+abs(x)) and x>=-mp.mpf('1e-60')))
                row['cells'].append(cell);record(dict(stage='cell_result',i=i,j=j,cell=cell))
                if not cell['passed']:save();raise AssertionError((i,j,k))
            with mp.workdps(160):
                raw=sum((mp.mpf(x['value120']['value']) for x in row['cells']),mp.mpf(0));rr=sum((mp.mpf(x['reference']['value']) for x in row['cells']),mp.mpf(0))
                mu=mean['summary']['mean'];assert len(mu)==16
                cov=raw-mp.mpf(mu[i])*mp.mpf(mu[j]);row.update(raw_second=mp.nstr(raw,140),reference_raw_second=mp.nstr(rr,100),covariance=mp.nstr(cov,140),complete=True)
            record(dict(stage='pair_complete',i=i,j=j));save()
        # Comparison only to existing saved reference and approximation outputs.
        with np.load(P['sample_reference'],allow_pickle=False) as z:
            bm=z['block_means'].copy();br=z['block_raw_second'].copy();sample_cov=z['covariance'].copy()
        assert bm.shape==(8,4,16) and br.shape==(8,4,16,16) and sample_cov.shape==(4,16,16)
        assert np.isfinite(bm).all() and np.isfinite(br).all() and np.isfinite(sample_cov).all()
        closure=json.loads(Path(P['closure']).read_text());mix=json.loads(Path(P['mixture']).read_text())
        assert closure['complete'] and mix['complete'] and closure['reference_sha256']==P['pins'][P['sample_reference']]
        cr=[r for r in closure['rows'] if r['step']==3];mr=[r for r in mix['rows'] if r['step']==3];assert len(cr)==len(mr)==1
        baselines={name:r['covariance'] for name,r in cr[0]['methods'].items()};assert set(baselines)=={'inherited_gauss','exact_h2_then_inherited_gauss'}
        baselines['retained_sector_gaussian']=mr[0]['covariance'];assert all(len(a)==16 and all(len(r)==16 for r in a) for a in baselines.values())
        with mp.workdps(180):
            convert=lambda x:mpq(F(float(x)))
            for row in result['rows']:
                i,j=row['i'],row['j'];cov=mp.mpf(row['covariance'])
                blocks=[convert(br[b,2,i,j])-convert(bm[b,2,i])*convert(bm[b,2,j]) for b in range(8)]
                bmean=sum(blocks)/8;se=mp.sqrt(sum((v-bmean)**2 for v in blocks)/(7*8))
                row['diagnostic']=dict(saved_sample_covariance_hex=float(sample_cov[2,i,j]).hex(),
                    new_minus_saved_sample_covariance=mp.nstr(cov-convert(sample_cov[2,i,j]),100),
                    block_covariances=[mp.nstr(v,120) for v in blocks],descriptive_block_standard_error=mp.nstr(se,100),
                    baseline_covariances={name:dict(value_hex=float(a[i][j]).hex(),baseline_minus_new=mp.nstr(convert(a[i][j])-cov,100)) for name,a in baselines.items()},
                    block_semantics='Exact MP combination of saved binary floats, population moments, not a new sampled reference or rigorous confidence interval.')
            lookup={(r['i'],r['j']):mp.mpf(r['covariance']) for r in result['rows']}
            checks=[]
            for j in (1,15):
                minor=lookup[0,0]*lookup[j,j]-lookup[0,j]**2
                checks.append(dict(i=0,j=j,minor=mp.nstr(minor,120),passed=bool(lookup[0,0]>=-mp.mpf('1e-50') and lookup[j,j]>=-mp.mpf('1e-50') and minor>=-mp.mpf('1e-50'))))
            result['selected_covariance_checks']=checks
            assert all(r['passed'] for r in checks)
        assert sum(len(r['cells']) for r in result['rows'])==170
        result['complete']=True;result['all_pass']=all(x['passed'] for r in result['rows'] for x in r['cells']);record(dict(stage='pilot_complete',cells=170));save()
    except BaseException as e:result['failure']=repr(e);record(dict(stage='failure',exception=repr(e)));raise
    finally:save()
    print(json.dumps(dict(complete=result['complete'],cells=170,cpu=result['cpu'])))
if __name__=='__main__':main()
