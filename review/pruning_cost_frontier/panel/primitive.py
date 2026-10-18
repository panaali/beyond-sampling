"""Exact retained-value fractional oracle; no target data or numeric libraries."""
from fractions import Fraction as F

def rational(x):
    if isinstance(x,bool) or not isinstance(x,(str,int,F)):raise ValueError('exact rational input required')
    try:return F(x)
    except (ValueError,ZeroDivisionError):raise ValueError('invalid rational')

def frontier(errors,costs,total,q,bits=80):
    if not isinstance(errors,list) or not isinstance(costs,list) or len(errors)!=len(costs):raise ValueError('paired lists required')
    if isinstance(bits,bool) or not isinstance(bits,int) or not 1<=bits<=128:raise ValueError('precision range')
    e=list(map(rational,errors));c=list(map(rational,costs));O=rational(total);Q=rational(q)
    M=sum(c,F());G=O-M
    if any(z<0 for z in e+c) or G<=0 or Q<=0:raise ValueError('nonnegative rows, positive fixed cost and threshold required')
    order=sorted((i for i in range(len(e)) if c[i]>0),key=lambda i:(e[i]/c[i],i))
    def feasible(s,b):return b*b*(O-s)<=Q
    points=[{'saved':F(),'error':F(),'feasible':True}]
    s=b=F()
    for i in order:
        s+=c[i];b+=e[i];points.append({'saved':s,'error':b,'feasible':feasible(s,b)})
    last=max(i for i,z in enumerate(points) if z['feasible'])
    trace=[]
    if last==len(order):lo=hi=M;boundary=None
    else:
        z=points[last];lo=z['saved'];hi=points[last+1]['saved'];i=order[last];a=e[i]/c[i]
        boundary={'index':i,'base_saved':lo,'base_error':z['error'],'slope':a}
        for _ in range(bits):
            mid=(lo+hi)/2;err=z['error']+a*(mid-z['saved']);ok=feasible(mid,err)
            trace.append({'mid':mid,'error':err,'feasible':ok})
            if ok:lo=mid
            else:hi=mid
    methods={}
    for name,ids,fixed in [('error_fixed',sorted(range(len(e)),key=lambda i:(e[i],i)),True),('error_self',sorted(range(len(e)),key=lambda i:(e[i],i)),False),('ratio_self',order,False)]:
        b=s=F();best_len=0;best={'indices':[],'saved':F(),'error':F()};rows=[]
        for prefix_len,i in enumerate(ids,1):
            s+=c[i];b+=e[i]
            ok=b*b*(O if fixed else O-s)<=Q
            rows.append({'index':i,'saved':s,'error':b,'feasible':ok})
            if ok and s>best['saved']:best_len=prefix_len;best={'indices':[],'saved':s,'error':b}
        best['indices']=ids[:best_len]
        assert feasible(best['saved'],best['error'])
        methods[name]={'prefixes':rows,'best':best}
    return {'errors':e,'costs':c,'total':O,'threshold':Q,'fixed_cost':G,'removable':M,'order':order,'breakpoints':points,'last_feasible_breakpoint':last,'boundary':boundary,'trace':trace,'fractional_saved_lower':lo,'fractional_saved_upper':hi,'methods':methods,'fractional_cost_lower':2*(O-hi)}
