from pathlib import Path
import json
import numpy as np
from instrument import Ledger,load

def main():
    rows=[]
    for name,a,b,expected in [('dot3',np.array([1.,2.,3.]),np.array([4.,5.,6.]),(3,2)),('mv4x3',np.arange(12.).reshape(4,3),np.arange(3.),(12,8)),('mm4x4x3',np.arange(16.).reshape(4,4),np.arange(12.).reshape(4,3),(48,36)),('zero_mv',np.zeros((4,3)),np.zeros(3),(12,8))]:
        L=Ledger();out=L.matmul(name,a,b);r=L.report();assert np.array_equal(out,a@b);assert (r['totals']['mul'],r['totals']['add'])==expected
        rows.append({'name':name,'ledger':r})
    mod,_,manifest=load(True);assert len(manifest['sites'])==18
    # Direct rows_at isolate the executed dense work even when every gate is off.
    model=(np.zeros(4),np.zeros((4,4)),np.zeros((3,4)));ops=mod.Ops();mod.rows_at(model,np.array([.5,-1.,2.]),ops);z=mod._ledger.report()
    assert z['totals']['historical_mac']==24
    assert z['totals']['mul']==168 and z['totals']['add']==123 and z['totals']['mask_mul']==36
    out={'cases':rows,'zero_rows_at':z,'zero_rows_at_historical_n':ops.n,'zero_rows_at_historical_x':getattr(ops,'x',0),'manifest':manifest,'scope':'primitive and named-kernel counts only'}
    Path(__file__).with_name('primitive_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'sites':len(manifest['sites']),'zero_rows_at':z['totals'],'historical_n':ops.n,'historical_x':getattr(ops,'x',0)}))
if __name__=='__main__':main()
