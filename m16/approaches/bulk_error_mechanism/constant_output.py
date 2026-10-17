"""Exact cone-interior constant argmax certificates, including boundary ties."""
import time
START=time.process_time()
import gzip,hashlib,json,resource
from collections import Counter,defaultdict
from fractions import Fraction as F
import numpy as np
import attribute as A
P=A.P

def rank(rows):
    rows=[list(map(F,row)) for row in rows];r=0
    for col in range(4):
        pivot=next((i for i in range(r,len(rows)) if rows[i][col]),None)
        if pivot is None:continue
        rows[r],rows[pivot]=rows[pivot],rows[r];v=rows[r][col];rows[r]=[x/v for x in rows[r]]
        for i in range(r+1,len(rows)):
            v=rows[i][col];rows[i]=[a-v*b for a,b in zip(rows[i],rows[r])]
        r+=1
        if r==len(rows):break
    return r

def finalmap(key,wi,wh,wo,D):
    mat=((0,)*4,)*4;scale=1
    for t in range(4):
        mat=tuple(tuple(sum(wh[i][j]*mat[j][q] for j in range(4))+(wi[i]*scale if q==t else 0) for q in range(4)) if key[4*t+i]=='1' else (0,)*4 for i in range(4));scale*=D
    return tuple(tuple(sum(row[i]*mat[i][q] for i in range(4)) for q in range(4)) for row in wo)

def main():
    stored=A.SOURCE.read_bytes();data=json.loads(gzip.decompress(stored));previous=json.loads((P/'attribution.json').read_text());f=lambda p:F(int(p[0]),int(p[1]));wi=list(map(f,data['model'][0]));wh=[list(map(f,row)) for row in data['model'][1]];wo=[list(map(f,row)) for row in data['model'][2]]
    D=max(v.denominator for v in wi+sum(wh,[])+sum(wo,[]));cv=lambda row:tuple(int(v*D) for v in row);wi=cv(wi);wh=list(map(cv,wh));wo=list(map(cv,wo))
    rays=defaultdict(set)
    for row,key in zip(data['gate_simplices'],previous['sets']['gate']['history_keys']):rays[key].update(tuple(map(int,r)) for r in row['integer_rays'])
    certs={};checks=0
    for key,vertices in rays.items():
        L=finalmap(key,wi,wh,wo,D);candidates=[];certificates={}
        for j in range(4):
            margins={};okay=True
            for k in range(4):
                if k==j:continue
                row=tuple(a-b for a,b in zip(L[j],L[k]));values=[A.dot(row,v) for v in vertices];checks+=len(values)
                pos=sum(v>0 for v in values);zero=sum(v==0 for v in values);neg=sum(v<0 for v in values)
                margins[k]=dict(positive=pos,zero=zero,negative=neg)
                if neg or (k<j and not pos):okay=False
            if okay:candidates.append(j);certificates[j]=margins
        assert len(candidates)<=1
        contrast_rank=rank([tuple(a-b for a,b in zip(row,L[0])) for row in L[1:]])
        old=previous['histories'][key]['group'];constant=bool(candidates)
        group=old if old!='remaining' else 'remaining_constant_output' if constant else 'remaining_decision_varying'
        certs[key]=dict(group=group,constant_label=candidates[0] if constant else None,vertex_count=len(vertices),contrast_rank=contrast_rank,certificate=certificates)
    result=dict(complete=False,certificates=certs,integer_margin_checks=checks,history_counts=dict(Counter(c['group'] for c in certs.values())),contrast_rank_counts=dict(Counter(c['contrast_rank'] for c in certs.values())),sets={},artifact_sha256=hashlib.sha256(stored).hexdigest(),prior_attribution_sha256=hashlib.sha256((P/'attribution.json').read_bytes()).hexdigest())
    for kind in ('gate','correct'):
        masses=defaultdict(float);bounds=defaultdict(float);labelmass=defaultdict(float)
        for row,key in zip(data[kind+'_simplices'],previous['sets'][kind]['history_keys']):
            normalized=[]
            for raw in row['integer_rays']:
                ray=list(map(int,raw));den=sum(abs(x) for x in ray);normalized.append([float(F(x,den)) for x in ray])
            value,error=A.V.mass(np.array(normalized).T,20);c=certs[key];masses[c['group']]+=value;bounds[c['group']]+=error
            if c['constant_label'] is not None:labelmass[c['constant_label']]+=value
        expected=previous['sets'][kind]['rules'][-1]['total'];assert abs(sum(masses.values())-expected)<1e-13
        result['sets'][kind]=dict(mass=dict(masses),conditional_bound=dict(bounds),constant_label_mass=dict(labelmass))
    result['error_by_group']={k:v-result['sets']['correct']['mass'].get(k,0) for k,v in result['sets']['gate']['mass'].items()}
    result.update(complete=True,cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);assert result['cpu_seconds_including_imports']<20
    (P/'constant_output.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='certificates'},indent=2))
if __name__=='__main__':main()
