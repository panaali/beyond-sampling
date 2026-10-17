"""Exact rank accuracy of the weight-derived last-input-only classifier."""
import numpy as np

def calculate(model):
    wi,_,wo=model;wi=np.asarray(wi,float).ravel();wo=np.asarray(wo,float);h=len(wi);t=len(wo)
    if not np.isfinite(wi).all() or not np.isfinite(wo).all():raise ValueError('finite weights required')
    plus=int(np.argmax(wo@np.maximum(wi,0)));minus=int(np.argmax(wo@np.maximum(-wi,0)))
    power=1.
    for _ in range(t):power*=2
    correction=(.5-(t+1)/power)/(t-1)
    value=1/t+((plus==t-1)-(minus==t-1))*correction
    # Input checks, 3H ReLU/negations, two TH dot products, two argmax,
    # T power multiplications, scalar rank formula and64 control allowance.
    operations=(h+t*h+10)+3*h+2*t*(2*h-1)+2*(t-1)+t+12+64
    return float(value),dict(positive_label=plus,negative_label=minus,scalar_operations=operations,
                            forward_equivalents=operations/(t*(2*h*h+h)+t*(2*h-1)),model_inputs_sampled=0)
