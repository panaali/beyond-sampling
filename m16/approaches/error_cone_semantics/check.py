"""Independent direct network witnesses; no cone engine or mass code imports."""
import time
START=time.process_time()
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import resource
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=Path('/workspace/work/recover-small_h2_910/m16/approaches/leaf_mass_review')
META=BASE/'error_capture.json'
DATA=BASE/'error_captured.npz'
WEIGHTS=Path('/workspace/work/t45_weights/2nd_argmax_8_4_1073741824_0.pth')
EXPECTED={META:'7cdd206f29c35dd5b46319b94d5ed5f5609e99873e9f0a6c73efe68fe9f48057',
          DATA:'15f9e5e483c69325c10fcbe4eaaa80f91503b177070ccec1ff2569a11c9436b6',
          WEIGHTS:'b771f40fcae826e010b2e89fb698e2ec35026119bc9f66d3ff046dba537b07da'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(name, result):
    result.update(cpu_seconds_including_imports=time.process_time()-START,
                  max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  script_sha256=sha(Path(__file__)))
    (HERE/name).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

def load():
    import torch
    torch.set_num_threads(1)
    for p,h in EXPECTED.items():assert sha(p)==h,str(p)
    meta=json.loads(META.read_text());assert meta['complete']
    data=np.load(DATA,allow_pickle=False)
    state=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    wi=state['rnn.weight_ih_l0'].numpy().astype(float)[:,0]
    wh=state['rnn.weight_hh_l0'].numpy().astype(float)
    wo=state['linear.weight'].numpy().astype(float)
    assert data['rays'].shape==(35098,4,4) and wi.shape==(8,) and wo.shape==(4,8)
    return meta,data,(wi,wh,wo)

def forward(x,model):
    wi,wh,wo=model; h=np.zeros((len(x),len(wi)))
    for t in range(x.shape[1]):h=np.maximum(0,h@wh.T+x[:,t,None]*wi)
    return h@wo.T

def exact_forward(x,model):
    wi,wh,wo=model;h=[F(0)]*len(wi);zeros=0;gate_signs=[]
    for v in x:
        z=[wi[i]*v+sum((wh[i][j]*h[j] for j in range(len(h))),F(0)) for i in range(len(h))]
        gate_signs.append([int(q>0)-int(q<0) for q in z]);zeros+=sum(q==0 for q in z)
        h=[max(F(0),q) for q in z]
    logits=[sum((row[j]*h[j] for j in range(len(h))),F(0)) for row in wo]
    return logits,zeros,gate_signs

def truth_label(x):return sorted(range(len(x)),key=lambda i:(-x[i],i))[1]
def prediction(logits):return max(range(len(logits)),key=lambda i:logits[i])
def rational_model(model):return ([F(float(v)) for v in model[0]],[[F(float(v)) for v in row] for row in model[1]],[[F(float(v)) for v in row] for row in model[2]])

def controls():
    cases=[
        # Independent hand calculations with recurrence h=max(0,h+x).
        dict(name='positive_recurrence',x=[3,1,2,-1],model=([1],[[1]],[[0],[1],[-1],[2]]),logits=[0,5,-5,10],truth=2,pred=3),
        dict(name='off_then_on',x=[-3,-1,2,-4],model=([1],[[1]],[[0],[1],[-1],[2]]),logits=[0,0,0,0],truth=1,pred=0),
        dict(name='duplicate_winning_rows',x=[3,1,2,-1],model=([1],[[0]],[[1],[1],[0],[-1]]),logits=[0,0,0,0],truth=2,pred=0),
        dict(name='nonzero_winning_tie',x=[-3,-1,-2,1],model=([1],[[0]],[[1],[1],[0],[-1]]),logits=[1,1,0,-1],truth=1,pred=0),
        dict(name='duplicate_hidden_rows',x=[3,1,2,1],model=([1,1],[[0,0],[0,0]],[[1,-1],[1,1],[-1,-1],[0,0]]),logits=[0,2,-2,0],truth=2,pred=1),
        dict(name='equal_input_rank_policy',x=[2,2,1,0],model=([0],[[0]],[[1],[0],[0],[0]]),logits=[0,0,0,0],truth=1,pred=0),
    ]
    rows=[]
    for c in cases:
        x=list(map(F,c['x']));m=rational_model(c['model']);y,z,g=exact_forward(x,m)
        yd=forward(np.array([c['x']],float),tuple(np.array(q,float) for q in c['model']))[0]
        assert y==list(map(F,c['logits'])) and np.array_equal(yd,c['logits'])
        assert truth_label(x)==c['truth'] and prediction(y)==c['pred']
        rows.append(dict(name=c['name'],pass_control=True,logits=list(map(str,y)),zeros=z))
    dump('controls.json',dict(controls=rows))
    print('controls passed',len(rows))

def floating():
    meta,data,model=load();rays=data['rays'];parent_ids=data['parent_ids']
    x=rays.sum(axis=1); logits=forward(x,model)
    order=np.argsort(-x,axis=1,kind='stable');truth=order[:,1]
    pred=np.argmax(logits,axis=1)
    parent_map={r['index']:r['truth'] for r in meta['parents']}
    expected=np.array([parent_map[int(i)] for i in parent_ids])
    competing=logits.copy();competing[np.arange(len(x)),truth]=-np.inf
    margin=logits[np.arange(len(x)),truth]-competing.max(axis=1)
    wrong_parent=truth!=expected;correct=pred==truth
    input_ties=np.any(np.diff(np.sort(x,axis=1),axis=1)==0,axis=1)
    failures=np.flatnonzero(wrong_parent|correct|input_ties).tolist()
    priority=failures[:64]
    for i in sorted(range(len(x)),key=lambda i:(abs(margin[i]),i)):
        if len(priority)==64:break
        if i not in priority:priority.append(i)
    np.savez_compressed(HERE/'floating.npz',inputs=x,logits=logits,truth=truth,prediction=pred,
                        expected_truth=expected,margin=margin,parent_ids=parent_ids,input_ties=input_ties)
    out=dict(complete=True,simplices=len(x),parents=len(meta['parents']),wrong_parent=np.flatnonzero(wrong_parent).tolist(),
             correctly_classified=np.flatnonzero(correct).tolist(),input_ties=np.flatnonzero(input_ties).tolist(),
             failure_count=len(failures),failure_count_beyond_exact_cap=max(0,len(failures)-64),exact_selection=priority,
             selected_margins=margin[priority].tolist(),minimum_absolute_margin=float(np.min(abs(margin))),
             source_hashes={str(p):sha(p) for p in EXPECTED},floating_archive_sha256=sha(HERE/'floating.npz'),
             random_network_inputs=0,forward_passes=len(x))
    dump('floating.json',out)
    print(json.dumps({k:out[k] for k in ('simplices','failure_count','minimum_absolute_margin','cpu_seconds_including_imports')}))

def exact_chunk(chunk):
    meta,data,model=load();model=rational_model(model)
    floating_result=json.loads((HERE/'floating.json').read_text());saved=np.load(HERE/'floating.npz',allow_pickle=False)
    assert sha(HERE/'floating.npz')==floating_result['floating_archive_sha256']
    selected=floating_result['exact_selection'][16*chunk:16*(chunk+1)];assert len(selected)==16
    rows=[]
    for i in selected:
        if time.process_time()-START>18:raise RuntimeError('exact chunk CPU cap; preserve completed checkpoint')
        r=data['rays'][i]; x=[sum((F(float(r[k,j])) for k in range(4)),F(0)) for j in range(4)]
        logits,zeros,gates=exact_forward(x,model);truth=truth_label(x);pred=prediction(logits)
        sorted_x=sorted(x,reverse=True); gaps=[sorted_x[j]-sorted_x[j+1] for j in range(3)]
        margin=logits[truth]-max(v for k,v in enumerate(logits) if k!=truth)
        row=dict(index=int(i),parent=int(data['parent_ids'][i]),truth=truth,prediction=pred,
                 expected_truth=int(saved['expected_truth'][i]),input_ties=any(g==0 for g in gaps),
                 pass_semantics=truth==int(saved['expected_truth'][i]) and pred!=truth and all(g>0 for g in gaps),
                 floating_truth=int(saved['truth'][i]),floating_prediction=int(saved['prediction'][i]),
                 exact_inputs=list(map(str,x)),exact_logits=list(map(str,logits)),rank_gaps=list(map(str,gaps)),
                 exact_margin=str(margin),margin_float=float(margin),zero_gate_count=zeros,gate_signs=gates)
        rows.append(row)
        dump(f'exact_{chunk}.json',dict(complete=len(rows)==16,chunk=chunk,rows=rows,
              floating_result_sha256=sha(HERE/'floating.json'),source_hashes={str(p):sha(p) for p in EXPECTED}))
    print(json.dumps(dict(chunk=chunk,checks=len(rows),failures=[r['index'] for r in rows if not r['pass_semantics']],cpu=time.process_time()-START)))

if __name__=='__main__':
    if sys.argv[1]=='exact':exact_chunk(int(sys.argv[2]))
    else:{'controls':controls,'floating':floating}[sys.argv[1]]()
