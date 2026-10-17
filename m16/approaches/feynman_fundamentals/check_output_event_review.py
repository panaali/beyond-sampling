"""Independent direct Gaussian selected-last rank integral; no author imports."""
import json, math, time
from pathlib import Path
from fractions import Fraction as F
from scipy.integrate import quad
from scipy.special import ndtr

start = time.process_time()
d = 12170831 / 67108864
m = 6
def integrand(u):
    f, q = float(ndtr(u)), float(ndtr(-u))
    g = f - .5
    # Exactly one earlier observation exceeds the selected last value:
    # one ordinary prefix, one of the two free positives, or constrained C.
    delta = float(ndtr(-u) - ndtr(-u/d))
    rank = m*f**(m-1)*q*g**3 + 2*f**m*g**2*q + f**m*g**2*delta
    return math.exp(-u*u/2)/math.sqrt(2*math.pi)*rank

pieces = [quad(integrand, a, b, epsabs=2e-15, epsrel=2e-13)
          for a,b in [(0,.25),(.25,1),(1,3),(3,9)]]
value = sum(x for x,e in pieces)
# The actual rank/event probability is <=1, so the omitted u>9 mass
# is bounded by the Gaussian tail; this does not certify quadrature rounding.
tail = float(ndtr(-9))

# Independent T107 analytic controls in CDF coordinates, exact fractions.
def monomial(power, lo, hi):
    return (hi**(power+1)-lo**(power+1))/(power+1)
z,h,o = F(0), F(1,2), F(1)
last_positive = 4*(monomial(3,h,o)-monomial(4,h,o))
first_positive = F(3,2)*(monomial(2,h,o)-monomial(3,h,o))
first_negative = F(7,2)*monomial(3,z,h)-4*monomial(4,z,h)
native_accuracy = last_positive + first_positive + first_negative
assert native_accuracy == F(89,320)
out = {'selected_last_joint':value,'adaptive_estimated_error':sum(e for x,e in pieces),
       'gaussian_tail_bound':tail,'event_probability':.25*(.25-math.atan(d)/(2*math.pi)),
       'native_last_positive':str(last_positive),'native_first_positive':str(first_positive),
       'native_first_negative':str(first_negative),'native_exact_accuracy':str(native_accuracy),
       'cpu_seconds':time.process_time()-start}
Path(__file__).with_name('output_event_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
