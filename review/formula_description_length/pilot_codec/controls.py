"""Fixed synthetic controls; no actual predictor or dataset evaluation."""
import ast, struct

def run(codec,rep,emit):
    fixtures=[None,False,True,-100000,0,100000,'αβ',[],[1,2],(1,'x'),0.0,-0.0,1.25,-.004,ast.parse('x = a + 2\ny = {0, 2, 7}')]
    for i,x in enumerate(fixtures):
        b=codec.encode(x);y=codec.decode(b);row={'kind':'roundtrip','index':i,'bits':b,'same':rep.same(x,y),'measure':rep.measure(x)};emit(row)
        assert row['same'] and len(b)==row['measure']['bits']
    mutations=[(ast.parse('x={0,2}'),ast.parse('x={0,3}')),(ast.parse('x=a+b'),ast.parse('x=a*b')),([1],[1,2])]
    for i,(a,b) in enumerate(mutations):
        aa=codec.encode(a);bb=codec.encode(b);row={'kind':'mutation','index':i,'a':aa,'b':bb};emit(row);assert aa!=bb
        if i==2:assert len(aa)<len(bb)
    cases=[([],[]),([0,7],[]),([8],[9]),([9],[7,8,9]),([10],list(range(1,10))),([11],list(range(1,10))),([8,9],[7,8,9]),([0,8,10],list(range(1,10)))]
    for fs,expected in cases:
        actual=rep.dependencies(fs);emit({'kind':'dependency','families':fs,'actual':actual,'expected':expected});assert actual==expected
    for bad in [[-1],[12],[True],[1.0]]:
        error=None
        try:rep.dependencies(bad)
        except ValueError as e:error=str(e)
        emit({'kind':'guard','input':bad,'error':error});assert error=='family'
    import numpy as np
    # One independent exact route/pooled and signed-zero table control; no RNN.
    env={'np':np};exec(compile(ast.parse((rep.P/'inference.py').read_text()),'synthetic-wrapper','exec'),env)
    c=np.array([2,3,9],dtype=np.int64);v=np.array([[1.,2.],[3.,4.],[5.,6.]],dtype=np.float64)
    record=([2,9],[[1.,0.],[0.,1.]],[[[2.,0.],[0.,3.]],[[0.,1.],[1.,0.]]])
    actual=env['table_predict'](c,v,record);expected=[[2.,6.],[3.,4.],[6.,5.]]
    emit({'kind':'route','actual':actual.tolist(),'expected':expected});assert np.array_equal(actual,np.asarray(expected))
    # Fixed codec framing rejection: valid prefix followed by one extra bit.
    error=None
    try:codec.decode(codec.encode(None)+'0')
    except AssertionError:error='trailing bit rejected'
    emit({'kind':'guard','input':'trailing bit','error':error});assert error
    return {'roundtrips':15,'mutations':3,'dependencies':8,'family_guards':4,'routes':1,'frame_guards':1,'rows':32,'table_calls':1,'actual_predictor_calls':0}
