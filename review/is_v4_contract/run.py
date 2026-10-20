#!/usr/bin/env python3
"""T462: frozen independent analytic controls; no network/model execution."""
import ast, csv, gzip, hashlib, json, math, os, resource, subprocess, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import integrate, optimize
from scipy.special import ndtr, ndtri, log_ndtr, logsumexp, xlogy
from scipy.stats import norm, multivariate_normal

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REV = 'a52da0d6a57d29aa2f76daa74c3360222f33aec2'
START = time.process_time()
R, N = 2000, 100000
HASHES = {}

def source(path):
    s = subprocess.check_output(['git', 'show', f'{REV}:{path}'], cwd=ROOT).decode()
    HASHES[path] = hashlib.sha256(s.encode()).hexdigest()
    return ast.parse(s)

g = dict(np=np, integrate=integrate, ndtr=ndtr, ndtri=ndtri, log_ndtr=log_ndtr,
         AD=.2, AD2=.2, G_MAXW=.05, G_CAPSHARE=.2)
s3 = source('m16/approaches/is_defensive/is3.py')
s4 = source('m16/approaches/is_v4/is4.py')
for tree, names in [(s3, ['phi', 'Z', 'Defensive', 'Mix3']), (s4, ['Mix4'])]:
    nodes = [x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned source>', 'exec'), g)
est = next(x for x in s4.body if isinstance(x, ast.FunctionDef) and x.name == 'estimate')
stats = next(x for x in est.body if isinstance(x, ast.FunctionDef) and x.name == 'stats')
exec(compile(ast.Module(body=[stats], type_ignores=[]), '<pinned stats>', 'exec'), g)
# Preserve the literal source gate statements, replacing only the supplied cap.
gate_nodes, active = [], False
for x in est.body:
    if isinstance(x, ast.Assign) and any(isinstance(y, ast.Name) and y.id == 'g_maxw' for y in x.targets):
        active = True
    if active:
        gate_nodes.append(x)
    if isinstance(x, ast.Assign) and any(isinstance(y, ast.Name) and y.id == 'flag' for y in x.targets):
        break
wrapper = ast.parse('def source_gate(W2, cap, p):\n    pass\n').body[0]
wrapper.body = gate_nodes + ast.parse('return dict(flag=flag, maxshare=g_maxw, capshare=g_cap, capped=n_capped)').body
exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), '<pinned gate>', 'exec'), g)


def independent_Z(t):
    return integrate.quad(lambda v: 360*norm.pdf(v)*norm.cdf(v)**7*norm.sf(v+t)**2,
                          -np.inf, np.inf, epsabs=1e-16, epsrel=1e-12)[0]


def proportion_check(name, observed, n, expected):
    tol = 6*math.sqrt(expected*(1-expected)/n) + 1e-5
    return dict(name=name, observed=float(observed), expected=float(expected), tolerance=tol,
                pass_check=bool(abs(observed-expected) <= tol))


def sampler_checks():
    out=[]
    for i,t in enumerate([1.25,1.75,2.25]):
        d=g['Defensive'](t); z=independent_Z(t)
        X=d.sample(np.random.default_rng(462000+i),100000)
        V=np.sort(X,axis=1); v=V[:,-3]
        checks=[]
        for cut in [-2,-1.5,-1,-.5,0,.5,1,1.5,2]:
            ref=integrate.quad(lambda y: 360*norm.pdf(y)*norm.cdf(y)**7*norm.sf(y+t)**2,
                               -np.inf,cut,epsabs=1e-16,epsrel=1e-11)[0]/z
            checks.append(proportion_check('v3<='+str(cut), np.mean(v<=cut),len(v),ref))
        for pos in range(10):
            checks.append(proportion_check('argmax='+str(pos),np.mean(X.argmax(1)==pos),len(X),.1))
        top_u=norm.sf(V[:,-2:])/norm.sf(v+t)[:,None]
        low_u=norm.cdf(V[:,:7])/norm.cdf(v)[:,None]
        for name,U in [('tops',top_u),('lows',low_u)]:
            for cut in [.1,.25,.5,.75,.9]:
                checks.append(proportion_check(name+'_pit<='+str(cut),np.mean(U<=cut),U.size,cut))
        out.append(dict(t=t,Z_source=d.Z,Z_reference=z,relative_Z_error=abs(d.Z-z)/z,
                        gap_violations=int(np.sum(V[:,-2]-v<t)),finite=bool(np.isfinite(X).all()),checks=checks,
                        passed=bool(all(c['pass_check'] for c in checks) and abs(d.Z-z)/z<=1e-10 and
                                    np.all(V[:,-2]-v>=t) and np.isfinite(X).all())))
    return out


def mixture_check():
    d=g['Defensive'](1.75)
    means=[np.zeros(10),np.array([2,2,-1]+[0]*7)]
    covs=[np.eye(10),np.diag([.5,1.5]+[1]*8)]
    m=g['Mix4'](d,means,covs,[.3,.7],.2)
    X,comp=m.sample(np.random.default_rng(462010),100000)
    V=np.sort(X,axis=1); hit=V[:,-2]-V[:,-3]>=1.75; Y=X[hit]
    logphi=norm.logpdf(Y).sum(1)
    ref=logsumexp(np.stack([math.log(.2)+logphi-math.log(independent_Z(1.75))]+[
        math.log(.8*w)+multivariate_normal.logpdf(Y,mean=mu,cov=C)
        for w,mu,C in zip([.3,.7],means,covs)],axis=1),axis=1)
    actual=m.logq(Y)-5*math.log(2*math.pi)
    W=np.exp(-.5*(Y**2).sum(1)-m.logq(Y))
    p,se=g['stats'](W,len(X)); cap=d.Z/.2
    diff=float(np.max(abs(actual-ref)))
    return dict(draws=len(X),gap_hits=len(Y),max_log_density_error=diff,weight_max=float(W.max()),cap=cap,
                p=float(p),se=float(se),Z=independent_Z(1.75),z=float((p-independent_Z(1.75))/se),
                passed=bool(diff<=1e-10 and W.max()<=cap*(1+1e-12) and abs(p-independent_Z(1.75))<=6*se))


def wilson(k,n):
    z=1.959963984540054; x=k/n; den=1+z*z/n
    c=(x+z*z/(2*n))/den; r=z*math.sqrt(x*(1-x)/n+z*z/(4*n*n))/den
    return [c-r,c+r]


def kl_interval(p,n,cap):
    x=min(max(p/cap,0.),1.); target=math.log(40)/n
    def f(y):
        return float(xlogy(x,x/y) + xlogy(1-x,(1-x)/(1-y))-target)
    if x==0: return (0.,min(1.,cap*(-math.expm1(-target))))
    if x==1: return (cap*math.exp(-target),min(1.,cap))
    lo=optimize.brentq(f,np.finfo(float).tiny,x,xtol=1e-16)
    hi=optimize.brentq(f,x,1-np.finfo(float).eps,xtol=1e-16)
    return max(0.,cap*lo),min(1.,cap*hi)


def one_summary(counts,weights,n,cap,ptrue):
    # Independent algebra handles all zero draws. Literal source receives explicit hit weights.
    W=np.repeat(weights,counts)
    ps,ss=g['stats'](W,n); gate=g['source_gate'](W,cap,ps)
    s1=float(np.dot(counts,weights)); s2=float(np.dot(counts,weights**2))
    p=s1/n; se=math.sqrt(max(0.,(s2-n*p*p)/(n*(n-1))))
    maxshare=float(max((w for w,c in zip(weights,counts) if c),default=0)/s1) if s1 else 1.
    capped=int(sum(c for w,c in zip(weights,counts) if w>.5*cap))
    capshare=float(sum(w*c for w,c in zip(weights,counts) if w>.5*cap)/s1) if s1 else 1.
    flag=bool(maxshare>.05 or (capped<30 and capshare>.2) or p==0)
    assert math.isclose(ps,p,rel_tol=1e-12,abs_tol=1e-20)
    assert math.isclose(ss,se,rel_tol=1e-9,abs_tol=1e-19)
    assert flag==gate['flag'] and capped==gate['capped']
    return dict(p=p,se=se,n=n,flag=flag,covered=bool(abs(p-ptrue)<=2*se),
                silent_twofold=bool(p<.5*ptrue and not flag),maxshare=maxshare,capshare=capshare,capped=capped)


def aggregate(rows,ptrue,kl=False):
    ph=np.array([r['p'] for r in rows]); se=np.array([r['se'] for r in rows]); ns=np.array([r['n'] for r in rows]);
    cover=sum(r['covered'] for r in rows); flags=sum(r['flag'] for r in rows)
    silent=sum(not r['covered'] and not r['flag'] for r in rows)
    mse=float(np.mean((ph-ptrue)**2)); F=mse*float(ns.mean())/(ptrue*(1-ptrue))
    result=dict(replications=len(rows),mean_estimate_ratio=float(ph.mean()/ptrue),
                variance=float(ph.var(ddof=1)),mean_se_squared=float(np.mean(se**2)),
                variance_over_mean_se_squared=float(ph.var(ddof=1)/np.mean(se**2)),
                mse=mse,normal_coverage=cover/len(rows),normal_coverage_wilson95=wilson(cover,len(rows)),
                flag_rate=flags/len(rows),unflagged_noncoverage=silent/len(rows),
                unflagged_noncoverage_wilson95=wilson(silent,len(rows)),
                reliability_screen_failed=bool(wilson(silent,len(rows))[0]>.05),
                silent_twofold_underestimates=sum(r['silent_twofold'] for r in rows),
                extension_rate=float(np.mean(ns>N)),mean_draws=float(ns.mean()),F_draws=F,
                strict_pass=bool(F<=1),lenient_pass=bool(F<=10))
    if kl:
        kc=sum(r['kl_lo']<=ptrue<=r['kl_hi'] for r in rows)
        result.update(kl_coverage=kc/len(rows),kl_coverage_wilson95=wilson(kc,len(rows)),
                      kl_median_width_over_truth=float(np.median([r['kl_hi']-r['kl_lo'] for r in rows])/ptrue))
    return result


def control(name,frac,seed,writer):
    rng=np.random.default_rng(seed)
    ptrue=1e-4 if name=='low_hit' else 1e-6
    if name=='low_hit':
        probs=np.array([ptrue,0]);weights=np.array([1.,0.]);cap=5.
    elif name=='covered':
        probs=np.array([.8+.2*ptrue,0]);weights=np.array([ptrue/probs[0],0.]);cap=5.
    else:
        A=(1-frac)*ptrue;B=frac*ptrue
        probs=np.array([.8+.2*A,.2*B]);weights=np.array([A/probs[0],5.]);cap=5.
    counts=rng.multinomial(N,np.r_[probs,1-probs.sum()],size=R)[:,:2]
    discovery = rng.binomial(2*N,.5*frac*ptrue,size=R) if frac else None
    fixed=[]; stopped=[]
    for j,c in enumerate(counts):
        row=one_summary(c,weights,N,cap,ptrue);row['kl_lo'],row['kl_hi']=kl_interval(row['p'],N,1. if name=='low_hit' else cap)
        fixed.append(row)
        extend=row['p']==0 or row['se']/row['p']>.1
        if extend:
            cext=rng.multinomial(9*N,np.r_[probs,1-probs.sum()])[:2]
            ext=one_summary(c+cext,weights,10*N,cap,ptrue)
        else: ext=dict(row)
        stopped.append(ext)
        for mode,r in [('fixed',row),('optional',ext)]:
            writer.writerow(dict(case=name,replication=j,mode=mode,**{k:r.get(k,'') for k in writer.fieldnames if k not in ['case','replication','mode']}))
    result=dict(p=ptrue,missing_fraction=frac,proposal_event_probabilities=probs.tolist(),event_weights=weights.tolist(),
                fixed=aggregate(fixed,ptrue,True),optional=aggregate(stopped,ptrue))
    if discovery is not None:
        result.update(stage1_no_missing_discovery=float(np.mean(discovery==0)),
                      stage1_no_missing_discovery_theory=float((1-.5*frac*ptrue)**(2*N)),
                      stage2_no_missing_hits=float(np.mean(counts[:,1]==0)),
                      stage2_no_missing_hits_theory=float((1-.2*frac*ptrue)**N))
    return result


def main():
    report=dict(source_revision=REV,source_hashes=HASHES,seeds_frozen=True,model_calls=0,
                sampler=sampler_checks(),mixture=mixture_check(),controls={},skipped=[])
    fields=['case','replication','mode','p','se','n','flag','covered','silent_twofold','maxshare','capshare','capped','kl_lo','kl_hi']
    with gzip.open(OUT/'replications.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name,frac,seed in [('missing_10pct',.1,462100),('missing_50pct',.5,462101),('covered',0,462102),('low_hit',0,462103)]:
            if time.process_time()-START>570:
                report['skipped'].append(name);continue
            report['controls'][name]=control(name,frac,seed,writer)
    report['compute']=dict(cpu_seconds=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           sampler_draws=300000,mixture_draws=100000,model_calls=0,
                           count_simulation='8000 multinomial/binomial replications; counts exactly represent stated IID trials, not individually generated uniforms',
                           threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']})
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(sampler_pass=all(x['passed'] for x in report['sampler']),mixture=report['mixture'],
                          controls=report['controls'],compute=report['compute'],skipped=report['skipped']),indent=2))

if __name__=='__main__':main()
