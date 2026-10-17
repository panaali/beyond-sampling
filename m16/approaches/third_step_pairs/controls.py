"""Twenty-four fixed analytic/domain controls; no trained weights or inputs."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time,resource
import mpmath as mp
from primitive import prepare,evaluate,pack,mpq
HERE=Path(__file__).resolve().parent
I=[[F(int(i==j)) for j in range(3)] for i in range(3)]
Q=[[F(3,5),F(-4,5),F(0)],[F(4,5),F(3,5),F(0)],[F(0),F(0),F(1)]]
D=[[F(-1),F(0),F(0)],[F(0),F(1),F(0)],[F(0),F(0),F(-1)]]

def transform(normals,R):return [[sum(R[i][j]*n[j] for j in range(3)) for i in range(3)] for n in normals]
def fixtures():
    orth=I;square=[[1,0,1],[1,0,-1],[0,1,1],[0,1,-1]]
    rows=[]
    def add(name,normals,kind='orthant',rotation=I,**kw):rows.append(dict(name=name,normals=normals,kind=kind,rotation=rotation,**kw))
    add('orthant',orth)
    add('positive_duplicate',orth+[[2,0,0]])
    add('edge_redundancy',orth+[[1,1,0]])
    add('interior_redundancy',orth+[[1,1,1]])
    add('reflected_orthant',transform(orth,D),rotation=D)
    add('rotated_orthant',transform(orth,Q),rotation=Q)
    add('normal_scale',[[F(1,2**30),0,0],[0,2**30,0],[0,0,3]])
    add('square',square,'square')
    add('rotated_square',transform(square,Q),'square',rotation=Q)
    add('reflected_square',transform(square,D),'square',rotation=D)
    for name,c,s in [('wedge',F(3),F(4)),('wide',F(-3),F(4)),('narrow20',F(1),F(1,2**20)),('narrow60',F(1),F(1,2**60)),('wide20',F(-1),F(1,2**20)),('wide60',F(-1),F(1,2**60))]:
        add(name,[[0,1,0],[s,-c,0],[0,0,1]],'wedge',c=c,s=s)
    add('zero_cone',orth+[[-1,-1,-1]],'zero')
    add('lower_dimension',[[1,0,0],[-1,0,0],[0,1,0],[0,0,1]],'zero')
    add('rank2',orth[:2],'unsupported')
    add('rank1',orth[:1],'unsupported')
    add('rank0',[],'unsupported')
    add('nan_input',[[float('nan'),0,0]],'invalid',exception='ValueError')
    add('wrong_shape',[[1,0]],'invalid',exception='ValueError')
    add('infinite_input',[[0,float('inf'),0]],'invalid',exception='ValueError')
    assert len(rows)==24
    return rows

def reference(row):
    with mp.workdps(160):
        kind=row['kind'];M=mp.matrix(3,3)
        if kind=='zero':p=mp.mpf(0)
        elif kind=='orthant':
            p=mp.mpf(1)/8
            for i in range(3):
                for j in range(3):M[i,j]=p if i==j else 1/(4*mp.pi)
        elif kind=='square':
            p=mp.mpf(1)/12;d=(1-1/mp.sqrt(3))/(4*mp.pi)
            M[0,0]=M[1,1]=p+d;M[2,2]=p-2*d;M[0,1]=M[1,0]=1/(2*mp.pi*mp.sqrt(3))
        elif kind=='wedge':
            c=mpq(row['c']);s=mpq(row['s']);a=mp.atan2(s,c);norm=mp.sqrt(c*c+s*s);p=a/(4*mp.pi)
            sn2=2*c*s/(c*c+s*s);cs2=(c*c-s*s)/(c*c+s*s)
            M[0,0]=p+sn2/(8*mp.pi);M[1,1]=p-sn2/(8*mp.pi);M[2,2]=p
            M[0,1]=M[1,0]=(1-cs2)/(8*mp.pi);M[0,2]=M[2,0]=s/norm/(4*mp.pi);M[1,2]=M[2,1]=(1-c/norm)/(4*mp.pi)
        else:raise ValueError(kind)
        R=mp.matrix([[mpq(F(x)) for x in r] for r in row['rotation']]);M=R*M*R.T
        return dict(mass=mp.nstr(p,160),matrix=[[mp.nstr(M[i,j],160) for j in range(3)] for i in range(3)])

def main():
    out=HERE/'results.json';progress=HERE/'progress.jsonl';assert not out.exists() and not progress.exists()
    P=json.loads((HERE/'protocol.json').read_text());assert P['source_gate'],'Independent source gate required'
    for path,sha in P['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
    result=dict(complete=False,rows=[],protocol_sha256=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest());start=time.process_time()
    def record(x):
        with progress.open('a') as f:f.write(json.dumps(pack(x),allow_nan=False)+'\n')
    def save():
        result.update(cpu=time.process_time()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(pack(result),indent=2,allow_nan=False)+'\n');tmp.replace(out)
    try:
        for row in fixtures():
            record(dict(stage='attempt',name=row['name'],kind=row['kind']))
            item=dict(name=row['name'],kind=row['kind'])
            if row['kind']=='invalid':
                try:prepare(row['normals'])
                except (ValueError,TypeError) as e:item.update(exception=type(e).__name__,message=str(e),passed=type(e).__name__==row['exception'])
                else:item.update(passed=False,unexpected='accepted invalid input')
            else:
                t=time.process_time();cert=prepare(row['normals']);item.update(certificate=cert,geometry_cpu=time.process_time()-t)
                if row['kind']=='unsupported':item['passed']=cert['status']=='unsupported_lineality'
                else:
                    assert cert['status']==('measure_zero' if row['kind']=='zero' else 'pointed_full')
                    t=time.process_time();lo=evaluate(cert,80);hi=evaluate(cert,120);item['evaluation_cpu']=time.process_time()-t
                    t=time.process_time();ref=reference(row);item['reference_cpu']=time.process_time()-t
                    with mp.workdps(180):
                        values=lambda d:[mp.mpf(d['mass'])]+[mp.mpf(x) for r in d['matrix'] for x in r]
                        a,b,r=values(lo),values(hi),values(ref)
                        errors=[abs(x-y) for x,y in zip(b,r)];precision=[abs(x-y) for x,y in zip(a,b)]
                        refpass=all(e<=mp.mpf('1e-65')+mp.mpf('1e-60')*abs(y) for e,y in zip(errors,r))
                        precpass=all(e<=mp.mpf('1e-65')*(1+abs(y)) for e,y in zip(precision,b))
                        M=mp.matrix([[mp.mpf(x) for x in rr] for rr in hi['matrix']]);mass=mp.mpf(hi['mass'])
                        symmetry=max(abs(M[i,j]-M[j,i]) for i in range(3) for j in range(3));trace=abs(sum(M[i,i] for i in range(3))-3*mass)
                        minors=[M[i,i] for i in range(3)]+[M[i,i]*M[j,j]-M[i,j]*M[j,i] for i,j in [(0,1),(0,2),(1,2)]]+[mp.det(M)]
                        item.update(value80=lo,value120=hi,reference=ref,maximum_absolute_error=mp.nstr(max(errors),100),maximum_precision_difference=mp.nstr(max(precision),100),
                                    symmetry_error=mp.nstr(symmetry,100),trace_error=mp.nstr(trace,100),principal_minors=[mp.nstr(x,100) for x in minors],
                                    reference_pass=refpass,precision_pass=precpass,passed=bool(refpass and precpass and symmetry<=mp.mpf('1e-65') and trace<=mp.mpf('1e-65') and min(minors)>=-mp.mpf('1e-65')))
            result['rows'].append(item);record(dict(stage='result',row=item));save();assert item['passed'],row['name']
        result['complete']=True;result['all_pass']=all(r['passed'] for r in result['rows'])
    except BaseException as e:result['failure']=repr(e);record(dict(stage='failure',exception=repr(e)));raise
    finally:save()
    print(json.dumps(dict(complete=result['complete'],passed=result['all_pass'],rows=len(result['rows']),cpu=result['cpu'])))
if __name__=='__main__':main()
