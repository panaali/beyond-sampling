"""Unchanged frozen T75 interface, cost in inherited 72-op forward equivalents."""
from common import estimate,OPS_PER_SAMPLE
FIXED_COST=True
SUPPORTS=[(4,3)]
def est(model,budget,rng):
    result=estimate(model)
    return result['value'],result['operations']/OPS_PER_SAMPLE
