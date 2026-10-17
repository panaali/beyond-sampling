"""T3 recurrent gate propagation in independently supported tie frames."""
from collections import defaultdict
import hashlib
import importlib.util
from pathlib import Path
import time
import numpy as np
from space import Space, unit

GEOMETRY = Path('/workspace/clean-neurons/m16/approaches/harmonic_sphere/cones.py')
spec = importlib.util.spec_from_file_location('t56_geometry', GEOMETRY)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Cone = module.Cone


def half_order_cells():
    eye = np.eye(3)
    for maximum in range(3):
        for target in range(3):
            if maximum == target:
                continue
            minimum = 3-maximum-target
            first = eye[maximum]-eye[target]
            second = eye[target]-eye[minimum]
            for near, far in ((first,second),(second,first)):
                a = unit(near)
                bisector = unit(far-near)
                b = unit(bisector-(bisector@a)*a)
                frame = np.stack((a,b,np.cross(a,b)),axis=1)
                yield target, a, bisector, frame


def estimate(model, normal_degree=8, tangent_degree=1, quadrature_order=None, seconds=100.):
    start = time.process_time()
    wi, wh, wo = (np.asarray(a,dtype=float) for a in model)
    H = len(wi)
    if wi.shape!=(H,) or wh.shape!=(H,H) or wo.shape!=(3,H):
        raise ValueError('requires T3 model')
    space = Space(normal_degree,tangent_degree,quadrature_order)
    work = defaultdict(float)
    # All twelve roots have the same bisector in their own frame.
    root = space.cut(space.one(),[-np.sqrt(3),1,0])
    root_total = 12*space.mass(root)
    if abs(root_total-1)>1e-8:
        raise ArithmeticError('root partition mass check failed')
    contributions = np.zeros(3)
    leaves = 0
    gate_partition = 0.
    def deadline():
        if time.process_time()-start>seconds:
            raise TimeoutError('anisotropic cycle CPU cap')
    def classify(cone, normal):
        value = cone.classify(normal)
        work['geometry_reported_ops'] += cone.flops
        cone.flops = 0.
        work['geometry_classifications'] += 1
        return value
    def add(cone, normal):
        cone.add(normal)
        work['geometry_reported_ops'] += cone.flops
        cone.flops = 0.
        work['geometry_additions'] += 1
    for target, a, bisector, frame in half_order_cells():
        cone = Cone(3)
        add(cone,a); add(cone,bisector)
        branches = [(cone,root.copy(),np.zeros((H,3)))]
        for t in range(3):
            next_branches = []
            for cone,c,A in branches:
                deadline()
                W = wh@A
                W[:,t] += wi
                work['gate_map_real_ops'] += 2*H*H*3+H
                live = [(cone,c,np.zeros(H,dtype=bool))]
                for j in range(H):
                    after_gate = []
                    if not np.any(W[j]):
                        continue
                    normal = unit(W[j])
                    work['plane_normalizations'] += 1
                    for K,g,mask in live:
                        side = classify(K,normal)
                        if side:
                            copied = mask.copy(); copied[j] = side>0
                            after_gate.append((K,g,copied))
                        else:
                            gp = space.cut(g,frame.T@normal)
                            work['frame_transform_real_ops'] += 18
                            Kp,Km = K.copy(),K.copy()
                            add(Kp,normal); add(Km,-normal)
                            mp,mm = mask.copy(),mask.copy(); mp[j]=True
                            after_gate.extend(((Kp,gp,mp),(Km,g-gp,mm)))
                    live = after_gate
                for K,g,mask in live:
                    next_branches.append((K,g,W*mask[:,None]))
                    work['gate_mask_products'] += H*3
            branches = next_branches
        leaves += len(branches)
        gate_partition += sum(space.mass(c) for _,c,_ in branches)
        for K,c,A in branches:
            deadline()
            L = wo@A
            work['readout_map_real_ops'] += 2*3*H*3
            accepted = True
            for other in range(3):
                if other==target:
                    continue
                margin = L[target]-L[other]
                work['readout_margin_subtractions'] += 3
                if not np.any(margin):
                    work['zero_readout_margins'] += 1
                    if other<target:
                        accepted=False;break
                    continue
                normal = unit(margin)
                work['plane_normalizations'] += 1
                side = classify(K,normal)
                if side<0:
                    accepted=False;break
                if side==0:
                    c = space.cut(c,frame.T@normal)
                    work['frame_transform_real_ops'] += 18
                    add(K,normal)
            if accepted:
                contributions[target] += space.mass(c)
    if abs(gate_partition-root_total)>1e-10:
        raise ArithmeticError('gate partition mass not conserved')
    # Explicit arithmetic ledger, not a certified total-operation bound:
    # complex multiply-add uses eight real ops; real multiply-add uses two.
    sw = dict(space.work)
    dense = (2*sw.get('real_gram_multiply_adds',0)
             +8*sw.get('complex_matvec_multiply_adds',0)
             +6*sw.get('complex_phase_products',0)
             +8*sw.get('radial_basis_values',0)
             +sw.get('scalar_transcendentals',0))
    arithmetic = dense+sum(v for k,v in work.items() if k.endswith('_ops') or k.endswith('_products') or k.endswith('_subtractions'))
    return float(contributions.sum()),dict(N=normal_degree,M=tangent_degree,
        quadrature_order=space.order,coefficients=space.size,leaves=leaves,
        root_mass=root_total,gate_partition_mass=gate_partition,
        contributions=contributions.tolist(),work=dict(work),cut_work=sw,
        accounted_real_arithmetic=arithmetic,
        accounting_limit='includes dense arithmetic and author geometry ledger; omits some indexing, comparisons and rule setup arithmetic; not a certified total FLOP bound',
        geometry_sha256=hashlib.sha256(GEOMETRY.read_bytes()).hexdigest(),
        cpu_seconds=time.process_time()-start,
        classification='deterministic supported spectral projection with numerical cut integrals',
        total_error_certified=False)
