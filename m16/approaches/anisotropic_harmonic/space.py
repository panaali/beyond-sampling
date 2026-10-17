"""Legendre/Fourier L2 sketch on a fixed positive hemisphere.

Whole-sphere mass is one half of the conditional constant coefficient.
Generic cuts use positive one-dimensional quadrature, not exact volumes.
"""
from collections import defaultdict
import numpy as np
from numpy.polynomial.legendre import leggauss, legvander


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    scale = np.max(abs(vector))
    if scale == 0:
        raise ValueError('zero normal needs explicit tie/gate handling')
    vector = vector/scale
    return vector/np.linalg.norm(vector)


class Space:
    def __init__(self, normal_degree, tangent_degree, quadrature_order=None):
        self.N = int(normal_degree)
        self.M = int(tangent_degree)
        self.nrad = self.N+1
        self.nang = 2*self.M+1
        self.size = self.nrad*self.nang
        self.constant = self.M
        self.order = quadrature_order or (4*self.N+32)
        self.work = defaultdict(float)
        x, w = leggauss(self.order)
        self.phi = (x+1)*np.pi/4
        self.weight = w*np.pi/4
        self.work['quadrature_rule_nodes_constructed'] += self.order
        self.eye = np.eye(self.size, dtype=complex)

    def one(self):
        c = np.zeros(self.size, dtype=complex)
        c[self.constant] = 1.
        return c

    def mass(self, c):
        if abs(c[self.constant].imag) > 1e-10:
            raise ArithmeticError('constant coefficient not real')
        return float(c[self.constant].real/2)

    def matrix(self, normal):
        """Normal is expressed in (tie-normal,tangent-b,tangent-c) frame."""
        v = unit(normal)
        self.work['normalized_cut_normals'] += 1
        az = v[0]
        r = np.hypot(v[1], v[2])
        if r == 0:
            return self.eye.copy() if az > 0 else np.zeros_like(self.eye)
        if az < 0:
            # Define the negative cut by complementation, avoiding duplicate integration.
            return self.eye-self.matrix(-v)
        delta = np.arctan2(v[2], v[1])
        modes = np.arange(-self.M, self.M+1)
        if az == 0:
            freq = modes[None, :]-modes[:, None]
            arc = np.empty(freq.shape, dtype=complex)
            arc[freq == 0] = .5
            nz = freq != 0
            arc[nz] = np.exp(1j*freq[nz]*delta)*np.sin(freq[nz]*np.pi/2)/(np.pi*freq[nz])
            self.work['angular_special_matrix_entries'] += self.nang**2
            return np.kron(np.eye(self.nrad), arc)
        # Complementary bad arc lies below z=r, centered at delta+pi.
        z = r*np.sin(self.phi)
        jac = r*np.cos(self.phi)
        denominator = np.sqrt(np.maximum(0., 1-z*z))
        ratio = az*np.sin(self.phi)/denominator
        if np.any(ratio > 1+1e-12):
            raise ArithmeticError('arc ratio out of domain')
        half_arc = np.arccos(np.clip(ratio, -1., 1.))
        radial = legvander(2*z-1, self.N)*np.sqrt(2*np.arange(self.nrad)+1)
        weighted = self.weight*jac
        blocks = []
        for k in range(2*self.M+1):
            angular = half_arc/np.pi if k == 0 else np.sin(k*half_arc)/(np.pi*k)
            blocks.append(radial.T @ ((weighted*angular)[:, None]*radial))
        bad = np.empty_like(self.eye)
        for i, mi in enumerate(modes):
            for j, mj in enumerate(modes):
                k = int(mj-mi)
                bad[i::self.nang, j::self.nang] = np.exp(1j*k*(delta+np.pi))*blocks[abs(k)]
        self.work['generic_cut_matrices'] += 1
        self.work['cut_quadrature_nodes'] += self.order
        self.work['radial_basis_values'] += self.order*self.nrad
        self.work['real_gram_multiply_adds'] += (2*self.M+1)*self.order*self.nrad**2
        self.work['complex_phase_products'] += self.size**2
        self.work['scalar_transcendentals'] += 4*self.order+(2*self.M)*self.order+self.nang**2
        return self.eye-bad

    def cut(self, coefficients, normal):
        matrix = self.matrix(normal)
        self.work['complex_matvec_multiply_adds'] += self.size**2
        return matrix @ coefficients
