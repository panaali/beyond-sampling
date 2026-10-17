"""Sufficient whole-gate correctness via conservative output/rank label sets."""
import time
START=time.process_time()
import gzip,hashlib,json,resource
from collections import Counter,defaultdict
from fractions import Fraction as F
import numpy as np
import constant_output as C
A=C.A;P=C.P

def main():
    stored=A.SOURCE.read_bytes();data=json.loads(gzip.decompress(stored));prior=json.loads((P/'attribution.json').read_text());f=lambda p:F(int(p[0]),int(p[1]));wi=list(map(f,data['model'][0]));wh=[list(map(f,row)) for row in data['model'][1]];wo=[list(map(f,row)) for row in data['model'][2]]
    D=max(v.denominator for v in wi+sum(wh,[])+sum(wo,[]));cv=lambda row:tuple(int(v*D) for v in row);wi=cv(wi);wh=list(map(cv,wh));wo=list(map(cv,wo))
    rays=defaultdict(set)
    for row,key in zip(data['gate_simplices'],prior['sets']['gate']['history_keys']):rays[key].update(tuple(map(int,r)) for r in row['integer_rays'])
    certs={};checks=0
    for key,vertices in rays.items():
        L=C.finalmap(key,wi,wh,wo,D);outputs=[];ranks=[];exclusions={};ranklogic={}
        for j in range(4):
            blockers=[];alwaysabove=[];possiblyabove=[]
            for k in range(4):
                if k==j:continue
                margin=tuple(a-b for a,b in zip(L[k],L[j]));values=[A.dot(margin,v) for v in vertices];checks+=len(values)
                if min(values)>=0 and (k<j or max(values)>0):blockers.append(k)
                d=[v[k]-v[j] for v in vertices];checks+=len(d)
                if min(d)>=0 and max(d)>0:alwaysabove.append(k)
                if max(d)>0:possiblyabove.append(k)
            if not blockers:outputs.append(j)
            exclusions[j]=blockers;ranklogic[j]=dict(always_above=alwaysabove,possibly_above=possiblyabove)
            if len(alwaysabove)<2 and possiblyabove:ranks.append(j)
        assert outputs and ranks
        if not set(outputs)&set(ranks):kind='uniformly_wrong'
        elif len(outputs)==len(ranks)==1 and outputs==ranks:kind='uniformly_correct'
        else:kind='unresolved'
        certs[key]=dict(kind=kind,possible_outputs=outputs,possible_second_rank=ranks,output_dominators=exclusions,rank_logic=ranklogic)
    result=dict(complete=False,certificates=certs,history_counts=dict(Counter(c['kind'] for c in certs.values())),integer_plane_checks=checks,sets={},artifact_sha256=hashlib.sha256(stored).hexdigest())
    for kind in ('gate','correct'):
        mass=defaultdict(float);bounds=defaultdict(float);counts=Counter()
        for row,key in zip(data[kind+'_simplices'],prior['sets'][kind]['history_keys']):
            name=certs[key]['kind'];assert kind!='correct' or name!='uniformly_wrong'
            vv=[]
            for raw in row['integer_rays']:
                r=list(map(int,raw));den=sum(abs(v) for v in r);vv.append([float(F(v,den)) for v in r])
            p,b=A.V.mass(np.array(vv).T,20);mass[name]+=p;bounds[name]+=b;counts[name]+=1
        result['sets'][kind]=dict(mass=dict(mass),conditional_bound=dict(bounds),counts=dict(counts))
        assert abs(sum(mass.values())-prior['sets'][kind]['rules'][-1]['total'])<1e-13
    gate=result['sets']['gate'];good=result['sets']['correct']
    assert abs(gate['mass'].get('uniformly_correct',0)-good['mass'].get('uniformly_correct',0))<=gate['conditional_bound'].get('uniformly_correct',0)+good['conditional_bound'].get('uniformly_correct',0)+1e-12
    lower=gate['mass'].get('uniformly_correct',0);upper=1-gate['mass'].get('uniformly_wrong',0)
    result.update(complete=True,accuracy_interval_ignoring_numerical_bounds=[lower,upper],unresolved_width=upper-lower,cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);assert result['cpu_seconds_including_imports']<20
    (P/'cone_correctness.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='certificates'},indent=2))
if __name__=='__main__':main()
