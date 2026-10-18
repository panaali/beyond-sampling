"""Fixed three stored exact models; weight predicates only, no network calls."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time,resource
from primitive import conditions,pack
H=Path(__file__).resolve().parent

def unpack_model(data):
    one=lambda z:F(int(z[0]),int(z[1]))
    return list(map(one,data[0])),[list(map(one,r)) for r in data[1]],[list(map(one,r)) for r in data[2]]

def main():
    out=H/'trained.json';progress=H/'trained_progress.jsonl';assert not out.exists() and not progress.exists()
    p=json.loads((H/'trained_protocol.json').read_text());assert p['source_gate']!='pending' and p['control_review']!='pending';start=time.process_time()
    def record(r):
        with progress.open('a') as f:f.write(json.dumps(pack(r),allow_nan=False)+'\n')
    result=dict(complete=False,rows=[],protocol_sha256=hashlib.sha256((H/'trained_protocol.json').read_bytes()).hexdigest())
    record(dict(stage='attempt',protocol_sha256=result['protocol_sha256']))
    try:
        for path,sha in p['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
        controls=json.loads((H/'controls.json').read_text());assert controls['complete'] and controls['all_pass'] and len(controls['rows'])==16
        for spec in p['models']:
            record(dict(stage='model_attempt',name=spec['name']));tick=time.process_time();archive=json.loads(Path(spec['path']).read_text());assert archive['complete']
            if spec['kind']=='capture':
                assert archive['T']==spec['T'] and archive['checkpoint_sha256']==spec['checkpoint_sha256']
                model=unpack_model(archive['model'])
            else:
                selected=[m for m in archive['models'] if m['name']==spec['name']];assert len(selected)==1
                row=selected[0];assert row['complete'] and row['checkpoint_sha256']==spec['checkpoint_sha256'];m=row['model']
                D=m['D'];assert isinstance(D,int) and D>0 and D&(D-1)==0 and m['H']==spec['H'] and m['T']==spec['T']
                model=([F(x,D) for x in m['wi']],[[F(x,D) for x in r] for r in m['wh']],[[F(x,D) for x in r] for r in m['wo']])
            Wi,Wh,Wo=model;assert len(Wi)==spec['H'] and len(Wo)==spec['T']
            decodes=len(Wi)+sum(map(len,Wh))+sum(map(len,Wo));load_cpu=time.process_time()-tick
            tick=time.process_time();answer=conditions(*model);predicate_cpu=time.process_time()-tick
            assert len(answer['subset_sums'])==spec['subsets']
            saved=dict(name=spec['name'],checkpoint_sha256=spec['checkpoint_sha256'],answer=answer,rational_weight_decodes=decodes,load_cpu=load_cpu,predicate_cpu=predicate_cpu)
            result['rows'].append(saved);record(dict(stage='model_result',row=saved))
        assert len(result['rows'])==3 and sum(len(r['answer']['subset_sums']) for r in result['rows'])==288
        result.update(complete=True,cpu_seconds=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      new_network_geometry_mass_reference_MC_calls=0,new_model_training=0,scope=p['scope'])
        record(dict(stage='complete',conditions=[r['answer']['checks'] for r in result['rows']]))
        out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(complete=True,cpu=result['cpu_seconds'],results=[dict(name=r['name'],conditions=r['answer']['checks'],identifiable=r['answer']['identifiable']) for r in result['rows']])))
    except BaseException as e:
        result.update(failure=repr(e),cpu_seconds=time.process_time()-start);record(dict(stage='failure',message=repr(e)));out.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');raise

if __name__=='__main__':main()
