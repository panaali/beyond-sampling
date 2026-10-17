// Optional serial C++ implementation of the cached affine reference walker.
// No fast-math: finite checks, signed comparisons, and exact ties are required.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

static double cdf(double x) { return .5 * std::erfc(-x / std::sqrt(2.)); }

extern "C" int walk_segments(int n, int H, int T, const double* Wi,
    const double* Wh, const double* Wo, const double* ys, const int64_t* aa,
    const int64_t* bb, const double* taus, int kind, double eps, int max_cells,
    double* masses, int64_t* cell_counts, int64_t* counters) {
  const double inf = std::numeric_limits<double>::infinity();
  const double invsq2 = 1. / std::sqrt(2.);
  // Allocation once per batch; only one line's state is resident.
  std::vector<double> B((T+1)*H), S((T+1)*H), roots(T), L0(T), L1(T);
  for (int row=0; row<n; ++row) {
    auto* w = counters + row*11;
    std::fill(w, w+11, 0);
    masses[row] = 0.; cell_counts[row] = 0;
    if (taus[row] == 0.) continue;
    std::fill(B.begin(), B.end(), 0.);
    std::fill(S.begin(), S.end(), 0.);
    std::fill(roots.begin(), roots.end(), inf);
    const double* y = ys + row*T;
    int a=aa[row], b=bb[row], first=std::min(a,b), restart=first;
    double at=-taus[row];
    for (int s=0; s<first; ++s) {
      for (int h=0; h<H; ++h) {
        double z=B[s*H]*Wh[h*H];
        for (int j=1; j<H; ++j) z += B[s*H+j]*Wh[h*H+j];
        z += y[s]*Wi[h];
        B[(s+1)*H+h]=std::max(z,0.);
      }
      w[0]++; w[6] += 2*H*H+2*H;
    }
    while (at < taus[row]) {
      if (cell_counts[row] >= max_cells) return 1;
      double tc=std::min(at+eps*(1.+std::abs(at)), std::nextafter(taus[row],at));
      for (int s=restart; s<T; ++s) {
        double next=inf, d=s==a ? invsq2 : (s==b ? -invsq2 : 0.);
        for (int h=0; h<H; ++h) {
          double z0=B[s*H]*Wh[h*H], z1=S[s*H]*Wh[h*H];
          for (int j=1; j<H; ++j) {
            z0 += B[s*H+j]*Wh[h*H+j]; z1 += S[s*H+j]*Wh[h*H+j];
          }
          z0 += y[s]*Wi[h]; z1 += d*Wi[h];
          bool gate=z0+tc*z1 > 0.;
          B[(s+1)*H+h]=gate ? z0 : 0.; S[(s+1)*H+h]=gate ? z1 : 0.;
          double r=-z0/z1;
          if (r>tc && std::isfinite(r)) next=std::min(next,r);
        }
        roots[s]=next;
        w[1]++; w[3]++; w[6] += 4*H*H+9*H; w[7] += 7*H-1;
      }
      int rstep=0;
      double root=inf;
      for (int s=0; s<T; ++s) if (roots[s]>tc && roots[s]<root) {root=roots[s];rstep=s;}
      double next=std::min(taus[row],root);
      if (!(next>at)) return 2;
      for (int k=0; k<T; ++k) {
        double v=B[T*H]*Wo[k*H], d=S[T*H]*Wo[k*H];
        for (int j=1; j<H; ++j) {v+=B[T*H+j]*Wo[k*H+j];d+=S[T*H+j]*Wo[k*H+j];}
        L0[k]=v; L1[k]=d;
      }
      w[2]+=2; w[8]+=2*T*(2*H-1); w[10]+=6*T+12; cell_counts[row]++;
      for (int side=0; side<2; ++side) {
        double lo=side==0 ? at : std::max(at,0.);
        double hi=side==0 ? std::min(next,0.) : next;
        if (!(hi>lo)) continue;
        int k=kind==23 ? (side==0 ? b : a) : (side==0 ? a : b);
        double clo=lo, chi=hi;
        bool bad=false;
        for (int j=0; j<T; ++j) {
          double c=L0[k]-L0[j], e=L1[k]-L1[j], r=-c/e;
          if (j!=k) {
            if (e>0.) clo=std::max(clo,r);
            if (e<0.) chi=std::min(chi,r);
          }
          if (e==0. && (c<0. || (c==0. && j<k))) bad=true;
        }
        double corr=cdf(chi)-cdf(clo), total=cdf(hi)-cdf(lo);
        if (!(chi>clo) || bad) corr=0.;
        masses[row]+=total-corr;
        w[4]++;w[5]+=4;w[9]+=20*T+12+4*50;
      }
      at=next;restart=rstep;
    }
  }
  return 0;
}
