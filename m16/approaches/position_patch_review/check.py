"""T26 bounded donor patches, matched controls and local linear feasibility."""
import time
START=time.process_time()
import argparse,hashlib,json,resource,itertools
from pathlib import Path
import numpy as np
import torch
MASKS=[64975,63951,65519,62927,65007]


def stats(x):
    x=np.asarray(x,dtype=float)
    if not len(x):return dict(n=0,mean=None,se=None)
    return dict(n=len(x),mean=float(x.mean()),se=float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path(__file__).with_name('results.json'));args=p.parse_args()
    sha=hashlib.sha256(args.weights.read_bytes()).hexdigest()
    assert sha=='498bd9a87c469e18e824a78e55c22843c7595776f1bfc2dfe07c36e5616218bb'
    state=torch.load(args.weights,weights_only=True,map_location='cpu')
    wi=state['rnn.weight_ih_l0'].numpy()[:,0].astype(float)
    wh=state['rnn.weight_hh_l0'].numpy().astype(float)
    wo=state['linear.weight'].numpy().astype(float)
    count=dict(gaussian_model_variates=0,gaussian_control_variates=0,hidden_steps=0,
        readout_vectors=0,gate_map_matrix_updates=0,eigendecompositions=0,
        control_qr=0,span_svd=0,least_squares=0,complement_svd=0,
        control_projection_vectors=0,partial_projection_vectors=0,witness_prefixes=0)

    def forward(xx,h=None):
        n,T=xx.shape;hh=np.zeros((n,T+1,16));hh[:,0]=0 if h is None else h
        gates=np.zeros((n,T,16),dtype=bool)
        for t in range(T):
            pre=hh[:,t]@wh.T+xx[:,t,None]*wi
            gates[:,t]=pre>0;hh[:,t+1]=np.maximum(pre,0)
        count['hidden_steps']+=n*T
        return hh,gates

    rng=np.random.default_rng(82026);rng.standard_normal((10000,10));x=rng.standard_normal((4096,10))
    count['gaussian_model_variates']=140960
    h,gate=forward(x);order=np.argsort(x,axis=1);m,s=order[:,-1],order[:,-2]
    code=gate[:,7].astype(int)@(1<<np.arange(16))
    use=(m<8)&(s<8)&(x.max(axis=1)>0)&np.isin(code,MASKS)
    original_ids=np.flatnonzero(use);x,h,gate,m,s,code=x[use],h[use],gate[use],m[use],s[use],code[use]
    n=len(x);assert n>0
    donor=x.copy();donor[np.arange(n),m],donor[np.arange(n),s]=x[np.arange(n),s],x[np.arange(n),m]
    hd,gd=forward(donor)
    base=h[:,8];delta=hd[:,8]-base
    amap=np.zeros((n,16,8))
    for t in range(8):
        amap=np.einsum('ij,njk->nik',wh,amap)
        amap[:,:,t]+=wi
        amap*=gate[:,t,:,None]
        count['gate_map_matrix_updates']+=n
    map_error=float(np.max(abs(np.einsum('nij,nj->ni',amap,x[:,:8])-base)))
    target_q=np.eye(16)[:,[3,13]]
    control_pairs=[pair for pair in itertools.combinations(range(16),2) if pair!=(3,13)]
    control_rng=np.random.default_rng(26001)
    fast={};control_spans={};mode_notes={}
    for c in MASKS:
        active=np.flatnonzero(c&(1<<np.arange(16)));ev,v=np.linalg.eig(wh[np.ix_(active,active)])
        count['eigendecompositions']+=1
        pairs=np.flatnonzero(ev.imag>1e-7);dominant=max(pairs,key=lambda j:abs(ev[j]))
        quicker=sorted([j for j in pairs if np.angle(ev[j])>np.angle(ev[dominant])+1e-8],key=lambda j:-abs(ev[j]))[:2]
        if len(quicker)<2:mode_notes[c]={'skip':'fewer than two faster pairs'};continue
        raw=np.column_stack([a for j in quicker for a in (v[:,j].real,v[:,j].imag)])
        u,sv,_=np.linalg.svd(raw,full_matrices=False);rank=int(np.sum(sv>max(sv[0],1)*1e-10));count['span_svd']+=1
        q=np.zeros((16,rank));q[active]=u[:,:rank];fast[c]=q
        controls=[]
        for k in range(64):
            raw=control_rng.standard_normal((len(active),rank));count['gaussian_control_variates']+=raw.size
            qq,_=np.linalg.qr(raw,mode='reduced');count['control_qr']+=1
            full=np.zeros((16,rank));full[active]=qq;controls.append(full)
        control_spans[c]=controls
        mode_notes[c]=dict(rank=rank,active=active.tolist(),fast_periods=[float(2*np.pi/np.angle(ev[j])) for j in quicker])
    logit_base=h[:,-1]@wo.T;logit_donor=hd[:,-1]@wo.T;count['readout_vectors']+=2*n
    contrast=lambda logits:logits[np.arange(len(logits)),m]-logits[np.arange(len(logits)),s]
    baseline_contrast=contrast(logit_base);donor_contrast=contrast(logit_donor)
    reports={};projection_records={}
    for kind in ['pair_3_13','fast_span']:
        dd=np.zeros_like(delta);chosen=np.zeros_like(delta);scores=np.full(n,np.inf);choices=np.full(n,-1)
        qs=[];selected_q=[];available=np.ones(n,dtype=bool)
        for i,c in enumerate(code):
            q=target_q if kind=='pair_3_13' else fast.get(int(c))
            if q is None:available[i]=False;qs.append(None);continue
            dd[i]=q@(q.T@delta[i]);qs.append(q)
        count['partial_projection_vectors']+=int(available.sum())
        target_energy=np.linalg.norm(dd,axis=1);target_readout=np.linalg.norm(dd@wo.T,axis=1)
        selected_q=[None]*n
        for c in MASKS:
            ids=np.flatnonzero((code==c)&available)
            controls=[np.eye(16)[:,pair] for pair in control_pairs] if kind=='pair_3_13' else control_spans.get(c,[])
            for j,q in enumerate(controls):
                dcontrol=(delta[ids]@q)@q.T
                energy=np.linalg.norm(dcontrol,axis=1);readout=np.linalg.norm(dcontrol@wo.T,axis=1)
                count['control_projection_vectors']+=len(ids)
                with np.errstate(divide='ignore',invalid='ignore'):
                    score=np.maximum(abs(np.log(energy/target_energy[ids])),abs(np.log(readout/target_readout[ids])))
                score=np.where(np.isfinite(score),score,np.inf)
                update=score<scores[ids];who=ids[update]
                scores[who]=score[update];choices[who]=j;chosen[who]=dcontrol[update]
                for i in who:selected_q[i]=q
        matched=available&(scores<=np.log(1.1));valid=available&np.isfinite(scores)
        patched=base+dd;controlled=base+chosen
        final,_=forward(x[:,8:],patched);final_control,_=forward(x[:,8:],controlled)
        logits=final[:,-1]@wo.T;control_logits=final_control[:,-1]@wo.T;count['readout_vectors']+=2*n
        effect=contrast(logits)-baseline_contrast;control_effect=contrast(control_logits)-baseline_contrast
        pred=logits.argmax(axis=1);cpred=control_logits.argmax(axis=1)

        def feasibility(displacements,qs):
            residuals=[];passing=[];ranks=[];witness_ids=[];witness_x=[]
            for i,dvec in enumerate(displacements):
                q=qs[i]
                if q is None:continue
                dx,_,_,_=np.linalg.lstsq(amap[i],dvec,rcond=1e-10);count['least_squares']+=1
                residual=float(np.linalg.norm(amap[i]@dx-dvec)/max(np.linalg.norm(dvec),1e-15))
                residuals.append(residual)
                comp=amap[i]-q@(q.T@amap[i]);sv=np.linalg.svd(comp,compute_uv=False);count['complement_svd']+=1
                ranks.append(int(np.sum(sv>max(sv[0],1)*1e-10)))
                if residual<=1e-10:
                    passing.append(i);witness_ids.append(i);witness_x.append(x[i,:8]+dx)
            witness_match=0;history_match=0;order_match=0
            if witness_ids:
                ids=np.array(witness_ids);wx=np.array(witness_x);hh,gg=forward(wx)
                count['witness_prefixes']+=len(ids)
                norms=np.linalg.norm(hh[:,-1]-(base[ids]+displacements[ids]),axis=1)
                witness_match=int(np.sum(norms<=1e-10*np.maximum(1,np.linalg.norm(base[ids]+displacements[ids],axis=1))))
                history_match=int(np.sum(np.all(gg==gate[ids,:8],axis=(1,2))))
                ww=x[ids].copy();ww[:,:8]=wx
                order_match=int(np.sum(np.all(np.argsort(ww,axis=1)==np.argsort(donor[ids],axis=1),axis=1)))
            return dict(tested=len(residuals),linear_pass=len(passing),numerical_forward_witness_pass=witness_match,
                same_strict_gate_history=history_match,donor_full_input_order_matches=order_match,
                residual_quantiles=list(map(float,np.quantile(residuals,[0,.25,.5,.75,1]))) if residuals else [],
                complement_rank_histogram={str(k):ranks.count(k) for k in sorted(set(ranks))})
        feasible_target=feasibility(dd,qs);feasible_control=feasibility(chosen,selected_q)
        reports[kind]=dict(available=int(available.sum()),matched=int(matched.sum()),unmatched=int(np.sum(available&~matched)),
            selected_control_scores=list(map(float,scores[valid])),
            all_available=dict(target_contrast_change=stats(effect[available]),control_contrast_change=stats(control_effect[available])),
            matched_only=dict(target_contrast_change=stats(effect[matched]),control_contrast_change=stats(control_effect[matched]),
                paired_target_minus_control=stats((effect-control_effect)[matched]),
                target_predicts_donor=stats((pred==m)[matched]),control_predicts_donor=stats((cpred==m)[matched]),
                full_donor_contrast_change=stats((donor_contrast-baseline_contrast)[matched])),
            negative_hidden_patch_count=int(np.sum(np.any(patched[available]<-1e-12,axis=1))),
            negative_hidden_control_count=int(np.sum(np.any(controlled[available]<-1e-12,axis=1))),
            target_local_feasibility=feasible_target,control_local_feasibility=feasible_control)
        projection_records[kind]=dict(selected_input_ids=original_ids.tolist(),available=available.tolist(),matched=matched.tolist(),
            chosen_control_index=choices.tolist(),target_contrast_change=effect.tolist(),control_contrast_change=control_effect.tolist())
    # Full donor positive control is globally reachable regardless of recipient cell.
    full_resume,_=forward(x[:,8:],hd[:,8]);full_error=float(np.max(abs(full_resume[:,-1]-hd[:,-1])))
    out=dict(checkpoint_sha256=sha,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection_count=n,selection_by_mask={str(c):int(np.sum(code==c)) for c in MASKS},
        positive_control=dict(recipient_accuracy=float(np.mean(logit_base.argmax(axis=1)==s)),
            donor_accuracy=float(np.mean(logit_donor.argmax(axis=1)==m)),
            full_donor_resume_max_error=full_error,gate_map_max_abs_error=map_error,
            donor_contrast_change=stats(donor_contrast-baseline_contrast)),
        mode_notes=mode_notes,reports=reports,paired_records=projection_records,counts=count,
        cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitations='Partial patches not assumed reachable. Local linear feasibility not global reachability; numerical witnesses not exact certificates. Matched controls selected using energy/readout norm only.')
    assert out['cpu_seconds_including_imports']<20
    args.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['selection_count','positive_control','counts','cpu_seconds_including_imports','max_rss_kib']},indent=2))
    for k,r in reports.items():print(k,json.dumps({z:r[z] for z in ['available','matched','matched_only','negative_hidden_patch_count','target_local_feasibility','control_local_feasibility']},indent=2))


if __name__=='__main__':main()
