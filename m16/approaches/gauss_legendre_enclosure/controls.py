"""Frozen exact arithmetic/root controls; no tensor or model evaluations."""
from pathlib import Path
import json,time,hashlib
from exact import *
HERE=Path(__file__).resolve().parent
def main():
    out=HERE/'controls.json';progress=HERE/'controls_progress.jsonl'
    assert not out.exists() and not progress.exists()
    P=json.loads((HERE/'protocol.json').read_text())
    for name,pin in P['source_pins'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==pin
    A=Arithmetic();rows=[];start=time.process_time()
    def case(name,actual,expected):
        row=dict(name=name,actual=pack(actual),expected=pack(expected),pass_=actual==expected);rows.append(row)
        with progress.open('a') as f:f.write(json.dumps(row)+'\n')
    for name,x,y,z in [('positive',(1,2),(3,4),(3,8)),('negative',(-2,-1),(3,5),(-10,-3)),('cross',(-3,2),(-4,5),(-15,12))]:
        case('multiply_'+name,imul(tuple(map(F,x)),tuple(map(F,y)),A),tuple(map(F,z)))
    for x,z in [((-2,-1),(1,4)),((-3,2),(0,9)),((1,2),(1,4))]:case('square_'+str(x),square(tuple(map(F,x)),A),tuple(map(F,z)))
    case('coeff2',coefficients(2,A),[F(-1,2),F(0),F(3,2)])
    case('coeff4',coefficients(4,A),[F(3,8),F(0),F(-30,8),F(0),F(35,8)])
    for n,points,expected in [(2,[-1,F(-1,2),0,F(1,2),1],[2,1,1,1,0]),(4,[-1,0,1],[4,2,0]),(6,[-1,0,1],[6,3,0])]:
        case('variations'+str(n),[signs_and_variations(n,F(x),A)[1] for x in points],expected)
    with progress.open('a') as f:f.write(json.dumps(dict(stage='attempt',name='two_point_rule'))+'\n')
    root=isolate(2,A);w=weight_interval(2,root['positive_roots'][0],A)
    case('two_point_rule',len(root['positive_roots'])==1 and not root['unresolved'] and w['dyadic'][0]<=1<=w['dyadic'][1] and all(r['contains'] for r in moments(2,root['positive_roots'],[w['dyadic']],A)),True)
    case('wide_derivative_unknown',weight_interval(2,(F(0),F(1,2)),A)['status'],'unknown_nonpositive_denominator')
    result=dict(cases=len(rows),failures=[r['name'] for r in rows if not r['pass_']],rows=rows,counts=dict(A.count),peak_bits=A.peak,cpu_seconds=time.process_time()-start)
    out.write_text(json.dumps(result,indent=2)+'\n');assert not result['failures']
if __name__=='__main__':
    try:main()
    except BaseException as e:
        with (HERE/'failures.jsonl').open('a') as f:f.write(json.dumps(dict(phase='controls',type=type(e).__name__,message=str(e)))+'\n')
        raise
