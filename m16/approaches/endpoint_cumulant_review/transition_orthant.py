"""Local frozen conditional-CDF transition knots; original bounded source retained."""
import inspect,hashlib
import full_covariance_bounded as B
source=inspect.getsource(B.orthant)
anchor='    if rho and lo < -b/rho < 12:cuts.append(float(-b/rho))'
assert source.count(anchor)==1
source=source.replace(anchor,'''    if rho:
        for t in [-12.,-8.,-4.,0.,4.,8.,12.]:
            z=(t*scale-b)/rho
            if lo<z<12:cuts.append(float(z))''')
space=dict(B.__dict__)
exec(compile(source,__file__+'[transition-knots]','exec'),space)
orthant=space['orthant']
source_hash=hashlib.sha256(inspect.getsource(B.orthant).encode()).hexdigest()
