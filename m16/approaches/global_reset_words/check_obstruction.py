from pathlib import Path
from fractions import Fraction as F
import json,time
from obstruction import four_word_witness,axis_preimage,direct

def strings(x):
    if isinstance(x,(tuple,list)):return [strings(v) for v in x]
    if isinstance(x,dict):return {k:strings(v) for k,v in x.items()}
    return str(x) if isinstance(x,F) else x

def main():
    start=time.process_time();old=json.loads(Path(__file__).with_name('pilot_results.json').read_text())
    words=[tuple(F(x) for x in r['word']) for r in old['cases'] if r['name'].startswith('four')]
    words.extend([(F(1,10**12),-10**12,F(1,3),F(-1,7)),(10**12,F(-1,10**12),-3,7),(-2,5,-11,13),(0,0,0,1),(1,-1,3,F(1,10**30))])
    rows=[]
    for name,wi,wh in (('trained',old['wi'],old['wh']),('fresh_rational',(-2,3),((2,-3),(4,-1)))):
        for word in words:
            h0,cert=four_word_witness(wi,wh,word);states=direct(wi,wh,h0,word)
            if states[0]!=cert['first_target'] or states[1]!=cert['second_target'] or states[2]!=(0,0) or states[3]==(0,0):raise AssertionError('constructed witness failed')
            rows.append(strings({'model':name,'word':word,'initial':h0,'states':states,'certificate':cert}))
        for word in ((0,0,0,0),(1,2,3,0)):
            try:four_word_witness(wi,wh,word)
            except ValueError:pass
            else:raise AssertionError('zero-last exclusion failed')
    bad=0
    for wh in (((1,-3),(4,-1)),((2,-1),(1,-1))):
        try:axis_preimage((-2,3),wh,1,2,3)
        except ValueError:bad+=1
    if bad!=2:raise AssertionError('parameter guard failed')
    result={'cases':rows,'cpu_seconds':time.process_time()-start,'four_word_witnesses':len(rows),'zero_last_exclusions':4,'invalid_model_exclusions':bad,'independent_of_polygon_enumeration':True,'sampled_inputs':0,'exact_forward_steps':4*len(rows)}
    Path(__file__).with_name('obstruction_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
if __name__=='__main__':main()
