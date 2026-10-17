"""Bounded fixed mass-selected gate/rank/readout feasibility panel."""
import time
START=time.process_time()
import gzip,hashlib,itertools,json,resource,warnings
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
warnings.filterwarnings('ignore',message='Unrecognized options detected')
import certify as C
import model as M
P=Path(__file__).resolve().parent;SOURCE=Path('/workspace/clean-neurons/m16/approaches/small_ladder_review/rational_geometry/geometry_h4_t4.json.gz')

def main(previous=None, output="panel.json"):
    stored=SOURCE.read_bytes();data=json.loads(gzip.decompress(stored));selection=json.loads((P/'selection.json').read_text());f=lambda p:F(int(p[0]),int(p[1]));wi=list(map(f,data['model'][0]));wh=[list(map(f,row)) for row in data['model'][1]];wo=[list(map(f,row)) for row in data['model'][2]]
    D=max(v.denominator for v in wi+sum(wh,[])+sum(wo,[]));cv=lambda row:tuple(int(v*D) for v in row);wi=cv(wi);wh=list(map(cv,wh));wo=list(map(cv,wo))
    result=dict(complete=False,selected_gate_mass=selection['selected_gate_mass'],selection_sha256=hashlib.sha256((P/'selection.json').read_bytes()).hexdigest(),histories=[],systems=[],counts={},random_model_inputs=0)
    if previous is not None:
        result=json.loads(Path(previous).read_text())
        result['prior_phase']=dict(lp_calls=result['lp_calls'],cpu_seconds=result['cpu_seconds_including_imports'],source_hashes=result['source_hashes'])
        result.pop('stop',None);result['complete']=False
    def save():
        result.update(lp_calls=C.CALLS,cumulative_lp_calls=C.CALLS+result.get('prior_phase',{}).get('lp_calls',0),cpu_seconds_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (P/output).write_text(json.dumps(result,indent=2)+'\n')
    def query(rows,history,top,second,kind,competitor=None):
        if rows is None:return dict(status='infeasible_semantic_tie',lp_calls=0)
        cert=C.feasible(rows)
        if cert['status']!='unknown':assert C.verify(rows,cert)
        if cert['status']=='feasible':
            x=list(map(F,cert['witness']));hist,pred=M.direct(wi,wh,wo,D,x);order=sorted(range(4),key=lambda j:x[j]);assert hist==history and order[-1]==top and order[-2]==second
            if kind=='correct':assert pred==second
            elif kind=='wrong':assert pred!=second
        index=len(result['systems']);result['systems'].append(dict(history=history,top=top,second=second,kind=kind,competitor=competitor,rows=[[str(v) for v in row] for row in rows],certificate=cert));return dict(status=cert['status'],system=index)
    try:
        for selected in selection['selected']:
            history=selected['history'];entry=next((h for h in result['histories'] if h['history']==history),None)
            if entry is not None and entry['complete']:continue
            planes,L=M.map_and_planes(history,wi,wh,wo,D)
            if entry is None:
                entry=dict(**selected,pairs=[],complete=False);result['histories'].append(entry)
            else:
                entry['restarted_partial_pairs']=[p for p in entry['pairs'] if 'classification' not in p]
                entry['pairs']=[p for p in entry['pairs'] if 'classification' in p]
            for top,second in itertools.permutations(range(4),2):
                if any(p['top']==top and p['second']==second for p in entry['pairs']):continue
                base=planes+M.rank_planes(top,second);pair=dict(top=top,second=second);entry['pairs'].append(pair)
                rank=query(base,history,top,second,'rank');pair['rank']=rank
                if rank['status']=='infeasible':pair['classification']='empty_rank';save();continue
                if rank['status']=='unknown':pair['classification']='unknown';save();continue
                extra=M.correct_planes(L,second);correct=query(None if extra is None else base+extra,history,top,second,'correct');pair['correct']=correct
                wrongs=[];pair['wrong_attempts']=wrongs
                for other in range(4):
                    if other==second:continue
                    extra=M.wrong_planes(L,second,other);wrong=query(None if extra is None else base+extra,history,top,second,'wrong',other);wrongs.append(dict(competitor=other,**wrong))
                    if wrong['status']=='feasible':break
                haswrong=any(v['status']=='feasible' for v in wrongs);hascorrect=correct['status']=='feasible';no_correct=correct['status'].startswith('infeasible');no_wrong=all(v['status'].startswith('infeasible') for v in wrongs)
                if hascorrect and haswrong:pair['classification']='mixed'
                elif no_correct and haswrong:pair['classification']='uniformly_wrong'
                elif hascorrect and no_wrong:pair['classification']='uniformly_correct'
                else:pair['classification']='unknown'
                save()
            entry['complete']=True
            print(history,dict(Counter(p.get('classification','partial') for p in entry['pairs'])),flush=True)
    except C.BudgetStop as e:result['stop']=str(e)
    result['complete']=all(h['complete'] for h in result['histories']) and len(result['histories'])==len(selection['selected'])
    statuses=Counter(p.get('classification','partial') for h in result['histories'] for p in h['pairs']);result['counts']=dict(statuses)
    mixed=[];uniform=[]
    for h in result['histories']:
        hasgood=any(p.get('correct',{}).get('status')=='feasible' for p in h['pairs']);hasbad=any(any(w['status']=='feasible' for w in p.get('wrong_attempts',[])) for p in h['pairs'])
        if hasgood and hasbad:mixed.append(h['history'])
        if h['complete'] and all(p['classification'] in ('empty_rank','uniformly_correct') for p in h['pairs']):uniform.append((h['history'],'correct'))
        if h['complete'] and all(p['classification'] in ('empty_rank','uniformly_wrong') for p in h['pairs']):uniform.append((h['history'],'wrong'))
    mass={h['history']:h['mass'] for h in result['histories']};result.update(exactly_witnessed_mixed_gate_histories=mixed,mixed_gate_mass=sum(mass[k] for k in mixed),whole_gate_uniform_certificates=uniform,whole_gate_resolved_mass=sum(mass[k] for k,_ in uniform),source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'certify.py',P/'model.py',P/'panel.py')});save();print({k:v for k,v in result.items() if k not in ['histories','systems','source_hashes']})
if __name__=='__main__':main()
