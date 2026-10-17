#include <cmath>
#include <cstdint>
#include <algorithm>
static constexpr double invsqrt2pi=0.398942280401432677939946059934;
static double Phi(double x){return 0.5*std::erfc(-x/std::sqrt(2.0));}
static void basis(double s,double t,double* v){v[0]=(1-s)*(1-t);v[1]=s*(1-t);v[2]=(1-s)*t;v[3]=s*t;}
extern "C" int gaussian_basis(std::int64_t n,const double* lo,const double* hi,
 const double* ss,const double* tt,const double* ds,const double* dt,
 const double* nodes,const double* weights,double* out,double* bounds,std::uint64_t* counts){
 for(std::int64_t i=0;i<n;i++){
  const double l=lo[i],r=hi[i],c=(l+r)/2,h=(r-l)/2;
  if(!(r>l)||!std::isfinite(l)||!std::isfinite(r)||!std::isfinite(ss[i])||!std::isfinite(tt[i])||!std::isfinite(ds[i])||!std::isfinite(dt[i]))return 1;
  if(l < -6.0 || r > 6.0)return 2;
  const double M=std::abs(c)*h+h*h/2;
  const double mass=l>=0?Phi(-l)-Phi(-r):Phi(r)-Phi(l);
  const double pl=std::exp(-l*l/2)*invsqrt2pi,pr=std::exp(-r*r/2)*invsqrt2pi;
  const double first=-pl*std::expm1(-2*c*h)-c*mass;
  const double second=(1+c*c)*mass-r*pl+l*pr;
  double vl[4],vc[4],vr[4];basis(ss[i]-ds[i],tt[i]-dt[i],vl);basis(ss[i],tt[i],vc);basis(ss[i]+ds[i],tt[i]+dt[i],vr);
  const bool badmom=mass<0||std::abs(first)>h*mass*(1+1e-12)||second<0||second>h*h*mass*(1+1e-12);
  bool badvalue=false;
  for(int k=0;k<4;k++){
   const double a=vc[k],b=(vr[k]-vl[k])/2,d=(vr[k]+vl[k])/2-vc[k];
   const double value=a*mass+b*(first/h)+d*(second/(h*h));out[4*i+k]=value;
   if(!std::isfinite(value)||value < -1e-12*mass||value>mass*(1+1e-12))badvalue=true;
  }
  const bool fallback=M<.125||badmom||badvalue;bounds[i]=0;
  counts[0]++;counts[2]+=badmom;counts[3]+=badvalue;
  if(fallback){
   counts[1]++;counts[4]+=12;
   for(int k=0;k<4;k++)out[4*i+k]=0;
   for(int j=0;j<12;j++){
    double v[4];basis(ss[i]+ds[i]*nodes[j],tt[i]+dt[i]*nodes[j],v);
    const double x=c+h*nodes[j],weight=h*weights[j]*std::exp(-x*x/2)*invsqrt2pi;
    for(int k=0;k<4;k++)out[4*i+k]+=weight*v[k];
   }
   const double m2=M*M,m4=m2*m2,m8=m4*m4,m11=m8*m2*M;
   bounds[i]=4*h*std::exp(-c*c/2)*invsqrt2pi*std::exp(M)*m11/39916800.;
  }
 }
 return 0;
}
