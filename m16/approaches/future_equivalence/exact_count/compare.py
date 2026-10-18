"""Saved-only exact/recipe and independent H4 history-prefix comparison."""
from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent

def histories(rows,same_truth=False):
    return {(r['truth'],tuple(tuple(g) for g in r['history'])) if same_truth else tuple(tuple(g) for g in r['history']) for r in rows}

def main():
    out=H/'full_comparison.json';progress=H/'comparison_progress.jsonl';assert not out.exists() and not progress.exists()
    p=json.loads((H/'full_protocol.json').read_text());assert p['source_gate']!='pending'
    result=dict(complete=False,new_geometry_forward_mass_calls=0)
    def record(x):
        with progress.open('a') as f:f.write(json.dumps(x)+'\n')
    record({'stage':'attempt'})
    try:
        for path,sha in p['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
        paths=[H/'full.json',Path(p['recipe_full']),Path(p['exact_h4_histories'])]
        exact,recipe,archive=[json.loads(q.read_text()) for q in paths]
        assert exact['complete'] and recipe['complete'] and archive['complete']
        result['pins']={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in paths}
        comparisons=[]
        for a,b in zip(exact['models'],recipe['models'],strict=True):
            assert a['H']==b['H'] and a['checkpoint_sha256']==b['checkpoint_sha256']
            row={'H':a['H'],'exact_summary':a['summary'],'recipe_summary':b['summary']}
            for same in (False,True):
                x,y=histories(a['rows'],same),histories(b['rows'],same);key='same_truth' if same else 'global'
                row[key]={'only_exact':sorted(x-y),'only_recipe':sorted(y-x),'intersection':len(x&y)}
            comparisons.append(row)
        keys=set(archive['sets']['gate']['history_keys']);assert len(keys)==450
        assert all(len(k)==16 and set(k)<=set('01') for k in keys)
        prefixes={tuple(tuple(map(int,k[j:j+4])) for j in (0,4,8)) for k in keys}
        h4=next(x for x in exact['models'] if x['H']==4);assert h4['checkpoint_sha256']==archive['checkpoint_sha256']
        captured=histories(h4['rows'])
        result.update(complete=True,comparisons=comparisons,h4_independent_archive={'full_histories':len(keys),'prefix_histories':len(prefixes),'captured_prefix_histories':len(captured),'only_archive':sorted(prefixes-captured),'only_capture':sorted(captured-prefixes)},scope='Saved history-set agreement only, not independent exact geometry coverage or mass proof')
        record({'stage':'complete'});out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['h4_independent_archive']))
    except BaseException as e:
        result['failure']=repr(e);record({'stage':'failure','failure':repr(e)});out.write_text(json.dumps(result,indent=2)+'\n');raise

if __name__=='__main__':main()
