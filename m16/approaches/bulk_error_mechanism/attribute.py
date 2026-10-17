"""Exact integer gate-history attribution of existing T68 simplices."""
import time
START=time.process_time()
import gzip,hashlib,importlib.util,json,math,resource
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;ROOT=Path('/workspace/clean-neurons');SOURCE=ROOT/'m16/approaches/small_ladder_review/rational_geometry/geometry_h4_t4.json.gz';MASS=ROOT/'m16/approaches/small_h4_4/stable_volume.py'
spec=importlib.util.spec_from_file_location('positive_reference_mass',MASS);V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
GROUPS=('final_all_off','earlier_all_off','contrast_forgetting_without_reset','remaining')
COUNT=Counter();HISTORIES={}
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def classify(rays,wi,wh,wo,D,correct):
    # Positive coefficients produce a cone-relative interior; avoid coincident input ranks.
    for powers in (range(1,5),(1,3,7,17),(1,5,29,173)):
        x=tuple(sum(w*r[q] for w,r in zip(powers,rays)) for q in range(4))
        if len(set(x))==4:break
    else:raise ArithmeticError('interior input-order coincidence')
    A=((0,)*4,)*4;h=(0,)*4;history=[];scale=1
    for t in range(4):
        B=tuple(tuple(sum(wh[i][j]*A[j][q] for j in range(4))+(wi[i]*scale if q==t else 0) for q in range(4)) for i in range(4))
        direct=tuple(dot(row,h)+wi[i]*x[t]*scale for i,row in enumerate(wh));mapped=tuple(dot(row,x) for row in B);assert direct==mapped
        gates=tuple(v>0 for v in direct);history.append(gates)
        for row,on in zip(B,gates):
            values=[dot(row,r) for r in rays];COUNT['exact_vertex_plane_checks']+=4
            assert all(v>=0 for v in values) if on else all(v<=0 for v in values)
        A=tuple(row if gates[i] else (0,)*4 for i,row in enumerate(B));h=tuple(max(v,0) for v in direct);scale*=D
    L=tuple(tuple(sum(row[i]*A[i][q] for i in range(4)) for q in range(4)) for row in wo)
    out=tuple(dot(row,h) for row in wo);assert out==tuple(dot(row,x) for row in L)
    prediction=max(range(4),key=lambda j:out[j]);target=sorted(range(4),key=lambda j:x[j])[-2]
    if correct:assert prediction==target
    key=''.join('1' if v else '0' for gs in history for v in gs)
    alloff=[t for t,g in enumerate(history) if not any(g)]
    hidden_forget=tuple(j for j in range(4) if all(row[j]==0 for row in A))
    contrast_forget=tuple(j for j in range(4) if all(row[j]==L[0][j] for row in L))
    ties=tuple((i,j) for i in range(4) for j in range(i) if L[i]==L[j])
    group=GROUPS[0] if 3 in alloff else GROUPS[1] if alloff else GROUPS[2] if contrast_forget else GROUPS[3]
    descriptor=dict(group=group,all_off_steps=alloff,hidden_forgotten=list(hidden_forget),contrast_forgotten=list(contrast_forget),identical_logit_pairs=ties)
    if key in HISTORIES:assert HISTORIES[key]==descriptor
    HISTORIES[key]=descriptor;COUNT['direct_recurrence_interiors']+=1
    return group,key

def main():
    stored=SOURCE.read_bytes();data=json.loads(gzip.decompress(stored));decode=lambda pair:F(int(pair[0]),int(pair[1]));wi=list(map(decode,data['model'][0]));wh=[list(map(decode,row)) for row in data['model'][1]];wo=[list(map(decode,row)) for row in data['model'][2]]
    D=max(x.denominator for x in wi+sum(wh,[])+sum(wo,[]));assert D&(D-1)==0
    conv=lambda row:tuple(int(x*D) for x in row);wi=conv(wi);wh=tuple(map(conv,wh));wo=tuple(map(conv,wo));out=dict(complete=False,source_sha256=hashlib.sha256(stored).hexdigest(),mass_source_sha256=hashlib.sha256(MASS.read_bytes()).hexdigest(),checkpoint_sha256=data['checkpoint_sha256'],groups=GROUPS,sets={})
    for kind in ('gate','correct'):
        tick=time.process_time();arrays=[];labels=[];histkeys=[];sums=defaultdict(F)
        for row in data[kind+'_simplices']:
            rays=[tuple(map(int,r)) for r in row['integer_rays']];group,key=classify(rays,wi,wh,wo,D,kind=='correct');labels.append(group);histkeys.append(key)
            normalized=[[float(F(v,sum(abs(x) for x in r))) for v in r] for r in rays];arrays.append(np.array(normalized).T)
            sums[group]+=F(*map(int,row['normalized_absolute_determinant']))
            assert time.process_time()-START<40
        result=dict(simplices=len(arrays),group_counts=dict(Counter(labels)),history_keys=histkeys,normalized_determinants_hex={k:[hex(v.numerator),hex(v.denominator)] for k,v in sums.items()},classification_cpu=time.process_time()-tick,rules=[]);out['sets'][kind]=result
        for n in (16,20):
            tick=time.process_time();prob=defaultdict(float);bound=defaultdict(float)
            for group,array in zip(labels,arrays):
                val,err=V.mass(array,n);prob[group]+=val;bound[group]+=err
            total=sum(prob.values());expected=1. if kind=='gate' else .477013125994119
            assert abs(total-expected)<=sum(bound.values())+1e-11
            result['rules'].append(dict(order=n,probability=dict(prob),conditional_integration_bound=dict(bound),total=total,cpu_seconds=time.process_time()-tick,nodes=len(arrays)*n**3))
            out.update(histories=HISTORIES,counts=COUNT,cpu_seconds_including_imports=time.process_time()-START);(P/'attribution.json').write_text(json.dumps(out,indent=2)+'\n')
            print(kind,n,total,dict(prob),flush=True);assert time.process_time()-START<40
    gate=out['sets']['gate']['rules'][-1];good=out['sets']['correct']['rules'][-1]
    out['error_by_group']={g:gate['probability'].get(g,0)-good['probability'].get(g,0) for g in GROUPS}
    out.update(complete=True,cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (P/'attribution.json').write_text(json.dumps(out,indent=2)+'\n');print(out['error_by_group'])
if __name__=='__main__':
    try:main()
    except Exception as e:
        import traceback
        (P/'attribution_failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),counts=COUNT,cpu_seconds=time.process_time()-START),indent=2)+'\n');raise
