"""Local observational wrapper for the pinned strat=True, cv=None SRB source.

The numerical AST and returned (estimate, cost) stay unchanged. Diagnostics
count completed RNG returns, never pretend to recover partial RNG internals.
"""
import ast
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class Snapshot:
    status: str
    quota: tuple
    requests: tuple
    requested_candidates: int
    completed_candidates: int
    source_drawn: object
    kept: object
    reported_cost: object
    error_type: object


class _RngProxy:
    def __init__(self, original, live):
        self.original, self.live = original, live

    def standard_normal(self, shape):
        # The pinned source calls only standard_normal((batch,9)). No resampling,
        # array copying, numeric transformation, seed access or extra RNG call.
        request = dict(requested=tuple(shape), returned=None, error=None)
        self.live['requests'].append(request)
        try:
            answer = self.original.standard_normal(shape)
        except BaseException as e:
            request['error'] = type(e).__name__
            raise
        request['returned'] = tuple(answer.shape)
        return answer


def numerical_module(source):
    root = ast.parse(source)
    functions = {'pstratum', 'pc', 'draw_all', 'make'}
    constants = {'T', 'EDGES', 'NS', '_pc', 'COST_DRAW'}
    body = [n for n in root.body if
            (isinstance(n, ast.FunctionDef) and n.name in functions) or
            (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets))]
    assert [n.name for n in body if isinstance(n, ast.FunctionDef)] == ['pstratum','pc','draw_all','make']
    assert {t.id for n in body if isinstance(n,ast.Assign) for t in n.targets} == constants
    # All selected nodes are original objects, unmodified, in original order.
    return ast.Module(body=body, type_ignores=[])


class ObservedEstimator:
    """Single-call-at-a-time local research adapter; latest snapshot is immutable.

    The accepted RNG ABI is exactly the source's ndarray-returning
    standard_normal((batch,9)). Arbitrary introspective RNG objects, threads,
    resource exhaustion and observer-allocation failures are outside parity.
    """
    def __init__(self, source, q_function, precompute, allocation):
        module = numerical_module(source)
        self.numeric_ast_sha256 = hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest()
        env = dict(np=np, Q=q_function, COST_Q=1.6, PRE=str(precompute))
        exec(compile(module,'<pinned-srb-observability>','exec'),env)
        self._env = env
        self._live = None
        self.last = None
        original_draw = env['draw_all']
        def observed_draw(nh,rng,batch=50_000):
            live=self._live
            if live is None: raise RuntimeError('draw outside active observation')
            live['quota']=tuple(int(v) for v in nh)
            value=original_draw(nh,rng,batch=batch)
            # Metadata only; original arrays/tuple are returned unchanged.
            groups, drawn=value
            live['source_drawn']=int(drawn)
            live['kept']=sum(len(x) for x in groups)
            return value
        env['draw_all']=observed_draw
        self._est=env['make'](strat=True,cv=None,alloc=allocation)

    def __call__(self,model,budget,rng):
        if self._live is not None: raise RuntimeError('nested observation unsupported')
        live=dict(quota=(),requests=[],source_drawn=None,kept=None,cost=None,status='running',error=None)
        self._live=live
        try:
            answer=self._est(model,budget,_RngProxy(rng,live))
            live.update(cost=answer[1],status='complete')
            return answer
        except BaseException as e:
            live.update(status='failure',error=type(e).__name__)
            raise
        finally:
            # Freeze copies of scalar/tuple metadata. No candidate arrays escape.
            records=tuple((v['requested'],v['returned'],v['error']) for v in live['requests'])
            self.last=Snapshot(live['status'],live['quota'],records,
                sum(x[0][0] for x in records),sum(x[1][0] for x in records if x[1] is not None),
                live['source_drawn'],live['kept'],live['cost'],live['error'])
            self._live=None
