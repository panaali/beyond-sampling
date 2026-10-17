#include <float.h>
#include <fenv.h>
#include <math.h>
#include <xmmintrin.h>

_Static_assert(FLT_RADIX == 2 && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 && DBL_MIN_EXP == -1021,
               "Requires binary64 double");

/* Status: fast0, exact fallback requested1, nonfinite2, environment3, internal4. */
int guard3(const double *a, double *out) {
    unsigned csr = _mm_getcsr();
    if (fegetround() != FE_TONEAREST || (csr & 0x8040u) != 0u ||
        (csr & 0x6000u) != 0u || (csr & 0x1f80u) != 0x1f80u) return 3;
    int finite = 1;
    for (int i=0; i<9; ++i) if (!isfinite(a[i])) finite = 0;
    if (!finite) return 2;
    double z[9];
    int eligible = 1;
    for (int i=0; i<9; ++i) {
        z[i] = fabs(a[i]);
        if (!(z[i] == 0.0 || (0x1p-100 <= z[i] && z[i] <= 0x1p100))) eligible = 0;
    }
    if (!eligible) return 1;

    double p0 = a[4]*a[8]; double p1 = a[5]*a[7];
    double m0 = p0-p1; double q0 = a[0]*m0;
    double p2 = a[3]*a[8]; double p3 = a[5]*a[6];
    double m1 = p2-p3; double q1 = a[1]*m1;
    double p4 = a[3]*a[7]; double p5 = a[4]*a[6];
    double m2 = p4-p5; double q2 = a[2]*m2;
    double r = q0-q1; double point = r+q2;

    double pp0 = z[4]*z[8]; double pp1 = z[5]*z[7];
    double pm0 = pp0+pp1; double pq0 = z[0]*pm0;
    double pp2 = z[3]*z[8]; double pp3 = z[5]*z[6];
    double pm1 = pp2+pp3; double pq1 = z[1]*pm1;
    double pp4 = z[3]*z[7]; double pp5 = z[4]*z[6];
    double pm2 = pp4+pp5; double pq2 = z[2]*pm2;
    double pr = pq0+pq1; double permanent = pr+pq2;
    double error = 0x1p-50*permanent;
    double lo, hi;
    if (error == 0.0) {
        if (point != 0.0) return 4;
        lo = hi = 0.0;
    } else {
        double below = point-error;
        double above = point+error;
        lo = nextafter(below, -INFINITY);
        hi = nextafter(above, INFINITY);
    }
    out[0] = point; out[1] = lo; out[2] = hi; out[3] = permanent; out[4] = error;
    return 0;
}

/* Explicit test-only environment accessors; guard3 never mutates caller modes. */
unsigned guard_getcsr(void) { return _mm_getcsr(); }
void guard_test_setcsr(unsigned value) { _mm_setcsr(value); }
int guard_getround(void) { return fegetround(); }
int guard_test_round(int which) {
    switch(which) {
        case 0: return fesetround(FE_TONEAREST);
        case 1: return fesetround(FE_DOWNWARD);
        case 2: return fesetround(FE_UPWARD);
        case 3: return fesetround(FE_TOWARDZERO);
        default: return -1;
    }
}
