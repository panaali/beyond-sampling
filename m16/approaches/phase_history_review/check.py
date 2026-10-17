"""Exact observed-history modal accounting; no causal or estimator claim."""
import time
START=time.process_time()
import argparse, hashlib, json, resource
from pathlib import Path
import numpy as np
import torch

MASKS=[64975,63951,65519,62927,65007]
SEEDS=[82026,82027]


def phase_summary(z,age,supported):
    unit=z/np.maximum(abs(z),1e-15)
    bins=[]
    for k in sorted(set(age.tolist())):
        values=unit[age==k]
        mean=values.mean()
        bins.append(dict(age=k,n=len(values),angle_degrees=float(np.angle(mean)*180/np.pi),
                         concentration=float(abs(mean)),small_amplitudes=int(np.sum(abs(z[age==k])<1e-15))))
    use=[b for b in bins if b['age'] in supported]
    slope=None
    if len(use)>=3:
        slope=float(np.polyfit([b['age'] for b in use],np.unwrap([b['angle_degrees']*np.pi/180 for b in use]),1)[0]*180/np.pi)
    return dict(slope_degrees=slope,bins=bins,small_amplitudes=int(np.sum(abs(z)<1e-15)))


def quantiles(v):
    return dict(zip(['min','p25','median','p75','p95','max'],map(float,np.quantile(v,[0,.25,.5,.75,.95,1]))))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=Path('/workspace/clean-neurons/m16/approaches/mechanism_checks'))
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('results.json'))
    args=parser.parse_args()
    digest=hashlib.sha256(args.weights.read_bytes()).hexdigest()
    assert digest=='498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb'
    state=torch.load(args.weights,weights_only=True,map_location='cpu')
    wi=state['rnn.weight_ih_l0'].numpy()[:,0].astype(float)
    wh=state['rnn.weight_hh_l0'].numpy().astype(float)
    count=dict(random_gaussian_variates=0,diagnostic_sequences=0,hidden_steps=0,
               eigendecompositions=0,matrix_inverses=0,selected_sequence_mask_pairs=0,
               recomputed_recurrent_vectors=0,complex_projected_vectors=0,
               exact_history_groups=0,reference_files_read=0)
    modes={}
    for code in MASKS:
        d=(code&(1<<np.arange(16)))!=0
        e,v=np.linalg.eig(wh[np.ix_(d,d)]);inv=np.linalg.inv(v)
        count['eigendecompositions']+=1;count['matrix_inverses']+=1
        ids=np.flatnonzero(e.imag>1e-7);j=max(ids,key=lambda i:abs(e[i]));lam=e[j]
        left=np.zeros(16,dtype=complex);left[d]=inv[j]
        astar=wh*d[:,None]*d[None,:]
        eigen_residual=float(np.max(abs(left@astar-lam*left)))
        h=np.arange(1,17,dtype=float)/16
        expected=left@h;max_unforced=0.
        for _ in range(10):
            h=astar@h;expected=lam*expected
            max_unforced=max(max_unforced,float(abs(left@h-expected)))
        modes[code]=(d,left,lam,dict(eigenrow_max_abs_residual=eigen_residual,
            unforced_linear_max_abs_residual=max_unforced,
            eigenbasis_condition=float(np.linalg.cond(v))))
    reports=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        rng.standard_normal((10000,10))
        x=rng.standard_normal((40000,10))
        count['random_gaussian_variates']+=500000
        hs=np.zeros((40000,11,16));codes=np.zeros((40000,10),dtype=int)
        for t in range(10):
            pre=hs[:,t]@wh.T+x[:,t,None]*wi
            hs[:,t+1]=np.maximum(pre,0)
            codes[:,t]=(pre>0).astype(int)@(1<<np.arange(16))
        count['diagnostic_sequences']+=len(x);count['hidden_steps']+=len(x)*10
        reference_path=args.reference/('results.json' if seed==82026 else 'results_seed82027.json')
        reference=json.loads(reference_path.read_text());count['reference_files_read']+=1
        ref_by_mask={z['gate_code']:z for z in reference['gate_checks']}
        rows=[]
        for code in MASKS:
            d,left,lam,controls=modes[code]
            selected=(codes[:,-1]==code)&(x.max(axis=1)>0)
            h,xx,cc=hs[selected],x[selected],codes[selected]
            n=len(h);count['selected_sequence_mask_pairs']+=n
            age=9-xx.argmax(axis=1);s=10-age
            support=[k for k in range(10) if np.sum(age==k)>=50]
            z=h@left;count['complex_projected_vectors']+=n*11
            initial=z[np.arange(n),s]
            base=lam**age*initial
            gate=np.zeros(n,dtype=complex);forcing=gate.copy();leak=gate.copy();output_gate=gate.copy()
            one_step_error=0.;state_error=0.
            for t in range(10):
                recurrent=h[:,t]@wh.T;dt=h[:,t+1]>0
                f=(dt*wi*xx[:,t,None])@left
                go=((dt.astype(int)-d.astype(int))*recurrent)@left
                gl=(h[:,t,~d]@wh[np.ix_(d,~d)].T)@left[d]
                count['recomputed_recurrent_vectors']+=n
                count['complex_projected_vectors']+=3*n
                g=go+gl
                one_step_error=max(one_step_error,float(np.max(abs(z[:,t+1]-lam*z[:,t]-g-f))))
                state_error=max(state_error,float(np.max(abs(h[:,t+1]-dt*(recurrent+xx[:,t,None]*wi)))))
                weight=lam**(9-t)*(t>=s)
                gate+=weight*g;forcing+=weight*f;output_gate+=weight*go;leak+=weight*gl
            full=base+gate+forcing;true=z[:,-1]
            absolute=abs(full-true)
            variants={name:phase_summary(val,age,support) for name,val in
                      [('observed',true),('initial',initial),('base',base),('base_gate',base+gate),
                       ('base_forcing',base+forcing),('full',full)]}
            reference_phase=ref_by_mask[code]['phase'] if 'phase' in ref_by_mask[code] else None
            original_count=ref_by_mask[code].get('n_final_positive_max')
            bins=variants['observed']['bins']
            expected_bins=reference_phase['age_bins'] if reference_phase else []
            count_match=original_count==n and [b['n'] for b in bins]==[b['n'] for b in expected_bins]
            ref_slope=reference_phase['observed_deg_per_age'] if reference_phase else None
            slope=variants['observed']['slope_degrees']
            slope_diff=None if slope is None or ref_slope is None else slope-ref_slope
            groups={}
            for i in range(n):
                key=tuple(int(q) for q in cc[i,s[i]:])
                groups.setdefault(key,[]).append(i)
            count['exact_history_groups']+=len(groups)
            grouped=[]
            for key,ids in groups.items():
                if len(ids)<20:continue
                ids=np.array(ids)
                ratio=true[ids]*np.conj(base[ids])
                u=ratio/np.maximum(abs(ratio),1e-15)
                grouped.append(dict(age=len(key),history=list(key),n=len(ids),
                    mean_phase_correction_degrees=float(np.angle(u.mean())*180/np.pi),
                    correction_concentration=float(abs(u.mean())),
                    gate_relative=quantiles(abs(gate[ids])/np.maximum(abs(true[ids]),1e-15)),
                    forcing_relative=quantiles(abs(forcing[ids])/np.maximum(abs(true[ids]),1e-15))))
            grouped.sort(key=lambda g:(-g['n'],g['history']))
            rows.append(dict(mask=code,n=n,age_support=support,lambda_real=float(lam.real),lambda_imag=float(lam.imag),
                predicted_degrees=float(np.angle(lam)*180/np.pi),controls=controls,phases=variants,
                original_e29_count_match=count_match,original_e29_slope_difference=slope_diff,
                one_step_identity_max_abs_error=one_step_error,state_recurrence_max_abs_error=state_error,
                final_reconstruction_max_abs_error=float(absolute.max()),
                final_reconstruction_max_relative_error=float(np.max(absolute/np.maximum(abs(true),1e-15))),
                component_to_final_amplitude={name:quantiles(abs(value)/np.maximum(abs(true),1e-15))
                    for name,value in [('base',base),('gate',gate),('forcing',forcing),('selected_output_gate',output_gate),('support_leakage',leak)]},
                exact_histories=len(groups),matched_groups_min20=grouped,
                matched_sequences_min20=sum(v['n'] for v in grouped)))
        reports.append(dict(seed=seed,reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),masks=rows))
        if time.process_time()-START>18:raise RuntimeError('Frozen CPU deadline reached; preserve scope, no expansion')
    out=dict(checkpoint_sha256=digest,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        left_eigenvector_convention='unconjugated row from inverse of right eigenvector matrix; positive-imaginary eigenvalue',
        reports=reports,counts=count,cpu_seconds_including_imports=time.process_time()-START,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitation='Observed histories and hidden states reused. No causal intervention, new estimator, or independent new cohort.')
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(cpu=out['cpu_seconds_including_imports'],counts=count,rows=[dict(seed=r['seed'],mask=q['mask'],n=q['n'],
        slopes={v:z['slope_degrees'] for v,z in q['phases'].items()},error=q['final_reconstruction_max_abs_error'],
        original_match=q['original_e29_count_match'],slope_difference=q['original_e29_slope_difference']) for r in reports for q in r['masks']]),indent=2))


if __name__=='__main__':main()
