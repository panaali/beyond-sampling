"""Frozen minimum-leakage reachable tangent rescue for T26."""
import time
START=time.process_time()
import argparse,hashlib,itertools,json,resource
from pathlib import Path
import numpy as np
import torch
from check import MASKS,stats


def summary(x):
    return dict(mean=float(np.mean(x)),quantiles=list(map(float,np.quantile(x,[0,.25,.5,.75,1]))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--previous',type=Path,default=Path(__file__).with_name('results.json'))
    p.add_argument('--out',type=Path,default=Path(__file__).with_name('reachable_results.json'));a=p.parse_args()
    old=json.loads(a.previous.read_text());state=torch.load(a.weights,weights_only=True,map_location='cpu')
    assert hashlib.sha256(a.weights.read_bytes()).hexdigest()==old['checkpoint_sha256']
    wi=state['rnn.weight_ih_l0'].numpy()[:,0].astype(float);wh=state['rnn.weight_hh_l0'].numpy().astype(float);wo=state['linear.weight'].numpy().astype(float)
    counts=dict(gaussian_model_variates=140960,gaussian_control_variates=0,hidden_steps=0,
                derivative_map_updates=0,eigendecompositions=0,span_svd=0,control_qr=0,
                reachability_svd=0,least_squares=0,readout_vectors=0)

    def forward(xx):
        nn=len(xx);h=np.zeros((nn,11,16));pre=np.zeros((nn,10,16))
        for t in range(10):
            pre[:,t]=h[:,t]@wh.T+xx[:,t,None]*wi;h[:,t+1]=np.maximum(pre[:,t],0)
        counts['hidden_steps']+=nn*10
        return h,pre

    rng=np.random.default_rng(82026);rng.standard_normal((10000,10));xx=rng.standard_normal((4096,10))
    ids=np.array(old['paired_records']['pair_3_13']['selected_input_ids']);x=xx[ids];n=len(x)
    h,pre=forward(x);g=pre>0;code=g[:,7].astype(int)@(1<<np.arange(16))
    order=np.argsort(x,axis=1);m,s=order[:,-1],order[:,-2]
    donor=x.copy();donor[np.arange(n),m],donor[np.arange(n),s]=x[np.arange(n),s],x[np.arange(n),m]
    hd,_=forward(donor);delta=hd[:,8]-h[:,8]
    amap=np.zeros((n,16,8));bmaps=np.zeros((n,10,16,8));a8=None
    for t in range(10):
        amap=np.einsum('ij,njk->nik',wh,amap)
        if t<8:amap[:,:,t]+=wi
        bmaps[:,t]=amap;amap=amap*g[:,t,:,None]
        if t==7:a8=amap.copy()
        counts['derivative_map_updates']+=n
    controls={};fast={};rr=np.random.default_rng(26001)
    for c in MASKS:
        active=np.flatnonzero(c&(1<<np.arange(16)));ev,v=np.linalg.eig(wh[np.ix_(active,active)]);counts['eigendecompositions']+=1
        pairs=np.flatnonzero(ev.imag>1e-7);dom=max(pairs,key=lambda j:abs(ev[j]));qq=sorted([j for j in pairs if np.angle(ev[j])>np.angle(ev[dom])+1e-8],key=lambda j:-abs(ev[j]))[:2]
        if len(qq)<2:continue
        raw=np.column_stack([z for j in qq for z in (v[:,j].real,v[:,j].imag)]);u,sv,_=np.linalg.svd(raw,full_matrices=False);counts['span_svd']+=1
        rank=int(np.sum(sv>max(sv[0],1)*1e-10));q=np.zeros((16,rank));q[active]=u[:,:rank];fast[c]=q;controls[c]=[]
        for _ in range(64):
            raw=rr.standard_normal((len(active),rank));counts['gaussian_control_variates']+=raw.size
            qc,_=np.linalg.qr(raw,mode='reduced');counts['control_qr']+=1
            full=np.zeros((16,rank));full[active]=qc;controls[c].append(full)
    coord=[pair for pair in itertools.combinations(range(16),2) if pair!=(3,13)]
    reports={}
    for kind in ['pair_3_13','fast_span']:
        saved=old['paired_records'][kind];ii=np.flatnonzero(saved['matched']);steps=[];disp=[];dxs=[];leaks=[];group_errors=[];failure=[]
        for i in ii:
            c=int(code[i]);which=saved['chosen_control_index'][i]
            target=np.eye(16)[:,[3,13]] if kind=='pair_3_13' else fast[c]
            control=np.eye(16)[:,coord[which]] if kind=='pair_3_13' else controls[c][which]
            u,sv,_=np.linalg.svd(a8[i],full_matrices=False);counts['reachability_svd']+=1
            u=u[:,sv>max(sv[0],1)*1e-10]
            pair_disp=[];pair_dx=[];pair_leak=[];pair_error=[];pair_limits=[]
            for q in [target,control]:
                b=q.T@delta[i]
                coef,_,_,_=np.linalg.lstsq(q.T@u,b,rcond=1e-10);counts['least_squares']+=1
                d=u@coef;dx,_,_,_=np.linalg.lstsq(a8[i],d,rcond=1e-10);counts['least_squares']+=1
                err=float(np.linalg.norm(q.T@d-b)/max(np.linalg.norm(b),1e-15));pair_error.append(err)
                pair_disp.append(d);pair_dx.append(dx)
                pair_leak.append(float(np.linalg.norm(d-q@(q.T@d))/max(np.linalg.norm(q.T@d),1e-15)))
                change=np.einsum('tij,j->ti',bmaps[i],dx)
                # For each nonzero original preactivation, a sign reversal first
                # occurs at -pre/change when these have opposite signs.
                toward=(pre[i]*change<0)
                limit=min(1.,float(np.min(-pre[i][toward]/change[toward])) if np.any(toward) else 1.)
                # Original zero preactivations belong to the inactive closed cell.
                if np.any((pre[i]==0)&(change>0)):limit=0.
                pair_limits.append(limit)
            alpha=.9*min(pair_limits)
            if max(pair_error)>1e-10:alpha=0.;failure.append(int(i))
            steps.append(alpha);disp.append(pair_disp);dxs.append(pair_dx);leaks.append(pair_leak);group_errors.append(pair_error)
        steps=np.array(steps);disp=np.array(disp);dxs=np.array(dxs);leaks=np.array(leaks);group_errors=np.array(group_errors)
        witnesses=[];errors=[];gate_changes=[];donor_orders=[];effects=[]
        for j in range(2):
            witness=x[ii].copy();witness[:,:8]+=steps[:,None]*dxs[:,j]
            hh,pp=forward(witness);predicted=h[ii,8]+steps[:,None]*disp[:,j]
            errors.append(float(np.max(abs(hh[:,8]-predicted))))
            gate_changes.append(int(np.sum(np.any((pp>0)!=g[ii],axis=(1,2)))))
            donor_orders.append(int(np.sum(np.all(np.argsort(witness,axis=1)==np.argsort(donor[ii],axis=1),axis=1))))
            logits=(hh[:,-1]-h[ii,-1])@wo.T;counts['readout_vectors']+=len(ii)
            effect=logits[np.arange(len(ii)),m[ii]]-logits[np.arange(len(ii)),s[ii]]
            effects.append(effect)
        energy=np.linalg.norm(disp,axis=2);readout=np.linalg.norm(disp@wo.T,axis=2)
        remaining_match=(np.maximum(abs(np.log(energy[:,0]/energy[:,1])),abs(np.log(readout[:,0]/readout[:,1])))<=np.log(1.1))
        positive=steps>0
        reports[kind]=dict(original_matched=len(ii),equality_failures=failure,positive_steps=int(positive.sum()),
            common_alpha=summary(steps),group_equality_max_relative_error=float(group_errors.max()),
            target_group_equality_max_relative_error=float(group_errors[:,0].max()),
            control_group_equality_max_relative_error=float(group_errors[:,1].max()),
            target_group_equality_failures=int(np.sum(group_errors[:,0]>1e-10)),
            control_group_equality_failures=int(np.sum(group_errors[:,1]>1e-10)),
            target_complement_to_group=summary(leaks[:,0]),control_complement_to_group=summary(leaks[:,1]),
            forward_hidden_max_abs_errors=errors,changed_gate_histories=gate_changes,
            donor_full_order_matches=donor_orders,surviving_energy_readout_matches=int(remaining_match.sum()),
            surviving_positive_energy_readout_matches=int(np.sum(remaining_match&positive)),
            target_contrast_change=stats(effects[0]),control_contrast_change=stats(effects[1]),
            paired_target_minus_control=stats(effects[0]-effects[1]),
            target_local_derivative=stats(effects[0][positive]/steps[positive]),
            control_local_derivative=stats(effects[1][positive]/steps[positive]),
            surviving_matched_paired_difference=stats((effects[0]-effects[1])[remaining_match]),
            source_input_ids=ids[ii].tolist(),alpha_per_pair=steps.tolist())
    out=dict(checkpoint_sha256=old['checkpoint_sha256'],first_phase_sha256=hashlib.sha256(a.previous.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reports=reports,counts=counts,
        cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Numerically witnessed within-cell perturbations preserving desired group coordinates up to common alpha, with unavoidable complementary leakage. Not isolated causal mediation or full donor swap.')
    assert out['cpu_seconds_including_imports']<20
    a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['counts','cpu_seconds_including_imports','max_rss_kib']},indent=2))
    for k,r in reports.items():print(k,json.dumps({key:v for key,v in r.items() if key not in ['source_input_ids','alpha_per_pair']},indent=2))


if __name__=='__main__':main()
