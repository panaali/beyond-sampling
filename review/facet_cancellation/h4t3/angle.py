"""Positive same-orthant trihedral Gaussian mass; no spherical clipping."""
from fractions import Fraction as F
from math import atan2, sqrt, pi, isfinite


def dot(a, b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]


def mass(row):
    rays = row['rays']
    if len(rays) != 3 or any(len(r) != 3 for r in rays):
        raise ValueError('three rays in three dimensions required')
    vertices = [[float(F(x, sum(map(abs, r)))) for x in r] for r in rays]
    a, b, c = vertices
    aa, bb, cc = dot(a,a), dot(b,b), dot(c,c)
    ab, bc, ca = dot(a,b), dot(b,c), dot(c,a)
    if min(aa,bb,cc) <= 0 or min(ab,bc,ca) < 0:
        raise ValueError('positive same-orthant kernel domain')
    na, nb, nc = sqrt(aa), sqrt(bb), sqrt(cc)
    denominator = ((na*nb*nc+ab*nc)+bc*na)+ca*nb
    determinant = float(F(*map(int, row['det'])))
    if not isfinite(denominator) or denominator <= 0 or not isfinite(determinant) or determinant <= 0:
        raise ValueError('positive finite solid-angle arguments')
    angle = atan2(determinant, denominator)
    unsigned = angle/(2*pi)
    value = row['weight']*unsigned
    if not isfinite(value):
        raise ArithmeticError('nonfinite solid-angle mass')
    return {'value_hex':value.hex(),'unsigned_hex':unsigned.hex(),'angle_hex':angle.hex(),
            'determinant_hex':determinant.hex(),'denominator_hex':denominator.hex(),
            'norm2_hex':[v.hex() for v in (aa,bb,cc)],'dot_hex':[v.hex() for v in (ab,bc,ca)],
            'weight':row['weight'],'normalized_vertices_hex':[[x.hex() for x in r] for r in vertices]}
