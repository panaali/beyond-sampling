"""Exact finite-law pairing identities, with explicit law and budget premises."""
from fractions import Fraction as F


def rational(q):
    if not isinstance(q, list) or len(q) != 2 or any(type(x) is not str or str(int(x)) != x for x in q):
        raise ValueError('canonical rational pair required')
    n, d = map(int, q)
    if d <= 0:
        raise ValueError('positive denominator required')
    return F(n, d)


def pair(q):
    return [str(q.numerator), str(q.denominator)]


def analyze(case):
    rows = case['rows']
    if not isinstance(rows, list) or not rows:
        raise ValueError('nonempty finite law required')
    states, weights, ys, ss = [], [], [], []
    for row in rows:
        x = row['x']
        if not isinstance(x, list) or not 2 <= len(x) <= 4 or any(type(v) is not int for v in x):
            raise ValueError('integer state of dimension 2..4 required')
        if states and len(x) != len(states[0]):
            raise ValueError('one dimension required')
        w, s = rational(row['weight']), rational(row['surrogate'])
        if w <= 0 or not 0 <= s <= 1 or type(row['correct']) is not int or row['correct'] not in (0, 1):
            raise ValueError('positive law weight, bounded surrogate and binary correctness required')
        states.append(tuple(x)); weights.append(w); ys.append(F(row['correct'])); ss.append(s)
    if len(set(states)) != len(states) or sum(weights) != 1:
        raise ValueError('distinct normalized law required')
    lookup = {x:i for i,x in enumerate(states)}
    if any(x[::-1] not in lookup for x in states):
        raise ValueError('reversal-closed support required')
    reverse = [lookup[x[::-1]] for x in states]
    invariant = all(weights[i] == weights[j] for i,j in enumerate(reverse))
    strict = case['require_invariant']
    if type(strict) is not bool or (strict and not invariant):
        raise ValueError('invariant law required by this request')
    single_cost, paired_cost, budget = map(rational, [case['single_cost'], case['paired_cost'], case['budget']])
    if min(single_cost, paired_cost, budget) <= 0:
        raise ValueError('positive declared costs and budget required')
    m, n = budget // single_cost, budget // paired_cost
    if min(m,n) < 1:
        raise ValueError('budget must fund at least one of each evaluation')
    residual = [y-s for y,s in zip(ys,ss,strict=True)]
    reversed_residual = [residual[j] for j in reverse]
    paired = [(a+b)/2 for a,b in zip(residual,reversed_residual,strict=True)]
    def mean(v):
        return sum((w*x for w,x in zip(weights,v,strict=True)), F())
    mu, mu_reverse, mu_pair = map(mean, [residual,reversed_residual,paired])
    variance = mean([(x-mu)**2 for x in residual])
    variance_reverse = mean([(x-mu_reverse)**2 for x in reversed_residual])
    variance_pair = mean([(x-mu_pair)**2 for x in paired])
    covariance = mean([(x-mu)*(y-mu_reverse) for x,y in zip(residual,reversed_residual,strict=True)])
    cancellation = mean([min(abs(x),abs(y)) if x*y < 0 else F() for x,y in zip(residual,reversed_residual,strict=True)])
    abs_residual, abs_reverse, abs_pair = map(mean, [list(map(abs,residual)),list(map(abs,reversed_residual)),list(map(abs,paired))])
    d = [i for i,x in enumerate(states) if x[0] < x[-1]]
    representatives = [i for i,x in enumerate(states) if x <= x[::-1]]
    orbit_mean = sum(((weights[i]*residual[i] if reverse[i] == i else weights[i]*residual[i]+weights[reverse[i]]*residual[reverse[i]]) for i in representatives), F())
    scalar = dict(accuracy=mean(ys),surrogate_mean=mean(ss),residual_mean=mu,
                  reversed_mean=mu_reverse,paired_mean=mu_pair,
                  surrogate_plus_pair=mean(ss)+mu_pair,absolute_residual=abs_residual,
                  absolute_reversed=abs_reverse,absolute_paired=abs_pair,cancellation=cancellation,
                  variance=variance,variance_reversed=variance_reverse,variance_paired=variance_pair,
                  covariance=covariance,orbit_mean=orbit_mean,
                  half_domain_mass=sum((weights[i] for i in d),F()),
                  half_domain_integral=sum((weights[i]*(residual[i]+reversed_residual[i]) for i in d),F()),
                  endpoint_tie_mass=sum((weights[i] for i,x in enumerate(states) if x[0]==x[-1]),F()),
                  fixed_point_mass=sum((weights[i] for i,j in enumerate(reverse) if i==j),F()))
    assert mean(ss)+mu == mean(ys) and orbit_mean == mu
    assert variance_pair == (variance+variance_reverse+2*covariance)/4
    assert (abs_residual+abs_reverse)/2-abs_pair == cancellation and abs_pair >= abs(mu_pair)
    if invariant:
        assert mu_pair == mu and abs_residual-abs_pair == cancellation
        assert variance_reverse == variance and variance_pair == (variance+covariance)/2
        if scalar['endpoint_tie_mass'] == 0:
            assert scalar['half_domain_mass'] == F(1,2) and scalar['half_domain_integral'] == mu
    return dict(invariant=invariant,scalars={k:pair(v) for k,v in scalar.items()},
                reverse=reverse,residuals=list(map(pair,residual)),paired=list(map(pair,paired)),
                representatives=representatives, single_count=m,paired_count=n,
                rho=pair(covariance/variance) if invariant and variance else None,
                budget_variance_ratio=pair(F(m,n)*variance_pair/variance) if variance else None,
                table_rows=len(rows),paired_residual_lookups=len(rows),
                model_evaluations_performed=0,declared_costs_are_hypothetical=True)


def labels(row):
    x, logits = row['x'], row['logits']
    if any(type(v) is not int for v in x+logits) or not 2 <= len(x) <= 4 or len(logits) != len(x):
        raise ValueError('matching integer input/logit vectors required')
    rank = lambda v: sorted(range(len(v)), key=lambda i:(-v[i],i))[1]
    truth, reversed_truth = rank(x), rank(x[::-1])
    winner = max(range(len(logits)), key=lambda i:logits[i])
    return dict(truth=truth,reversed_truth=reversed_truth,mapped_truth=len(x)-1-truth,
                relabeling_holds=reversed_truth==len(x)-1-truth,
                strict_ranks=len(set(x))==len(x),first_index_winner=winner)
