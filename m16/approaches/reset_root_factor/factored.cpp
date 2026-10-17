#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using Real=long double;
static const Real INF=std::numeric_limits<Real>::infinity();

// All pair roots split the interval, so its right-limit ordering is constant.
static int compare(Real a,Real b,Real c,Real d,Real lo,int unused,bool& invalid) {
    Real intercept=a-c,slope=b-d;
    if(!std::isfinite(intercept)||!std::isfinite(slope)){invalid=true;return 0;}
    if(slope==0)return (intercept>0)-(intercept<0);
    Real root=-intercept/slope;
    if(!std::isfinite(root)){invalid=true;return 0;}
    bool positive=slope>0?lo>=root:lo<root;
    return positive?1:-1;
}
static Real angle_width(Real lo,Real hi) {
    const Real halfpi=acosl(-1.L)/2;
    if(std::isinf(lo))return hi<=0?atan2l(1,-hi):atanl(hi)+halfpi;
    if(std::isinf(hi))return lo>=0?atan2l(1,lo):halfpi-atanl(lo);
    if(lo<0&&hi>0)return atanl(hi)-atanl(lo);
    Real x=lo>=0?lo:-hi,y=lo>=0?hi:-lo,scale=std::max(1.L,y);
    return atan2l((y-x)/scale,1/scale+x*(y/scale));
}

extern "C" int projective_walk(int n,int h,int t,const double* wi,const double* wh,
    const double* wo,const double* U,const double* V,int cap,double* answer,
    double* coverage,long long* counters) {
    const Real pi=acosl(-1.L);
    for(int sample=0;sample<n;++sample) {
        long long* count=counters+sample*9;
        for(int j=0;j<9;++j)count[j]=0;
        Real mass=0,full=0;
        for(int sign:{1,-1}) {
            std::vector<Real> input_a(t),input_b(t),input_roots;
            for(int j=0;j<t;++j){input_a[j]=sign*(Real)U[sample*t+j];input_b[j]=sign*(Real)V[sample*t+j];}
            for(int i=0;i<t;++i)for(int j=i+1;j<t;++j) {
                Real a=input_a[i]-input_a[j],b=input_b[i]-input_b[j];++count[2];
                if(b!=0){Real root=-a/b;if(!std::isfinite(root))return 3;input_roots.push_back(root);}
            }
            Real current=-INF;int restart=0;
            std::vector<Real> ha((t+1)*h,0),hb((t+1)*h,0),roots(t,INF);std::vector<unsigned char> zero(t+1,1);
            while(current<INF) {
                if(count[0]>=cap)return 1;
                ++count[0];
                for(int step=restart;step<t;++step) {
                    ++count[1];
                    roots[step]=INF;bool all_inactive=true;++count[7];if(step>0&&zero[step])++count[8];
                    for(int j=0;j<h;++j) {
                        Real a=(Real)wi[j]*input_a[step],b=(Real)wi[j]*input_b[step];
                        for(int k=0;k<h;++k){a+=(Real)wh[j*h+k]*ha[step*h+k];b+=(Real)wh[j*h+k]*hb[step*h+k];}
                        if(!std::isfinite(a)||!std::isfinite(b))return 3;
                        bool active=a>0;
                        ++count[2];
                        if(zero[step]) {
                            // A certified all-inactive prefix is identically zero; cancel Wi before division.
                            if(wi[j]!=0 && input_b[step]!=0) {
                                Real root=-input_a[step]/input_b[step];if(!std::isfinite(root))return 3;
                                bool increasing=(wi[j]>0)==(input_b[step]>0);
                                active=increasing?current>=root:current<root;
                                if(root>current&&root<roots[step])roots[step]=root;
                            } else active=(wi[j]>0&&input_a[step]>0)||(wi[j]<0&&input_a[step]<0);
                        } else if(b!=0) {
                            Real root=-a/b;if(!std::isfinite(root))return 3;
                            active=b>0?current>=root:current<root;
                            if(root>current&&root<roots[step])roots[step]=root;
                        }
                        all_inactive=all_inactive&&!active;ha[(step+1)*h+j]=active?a:0;hb[(step+1)*h+j]=active?b:0;
                    }
                    zero[step+1]=all_inactive;
                }
                Real end=INF;restart=0;
                for(int step=0;step<t;++step)if(roots[step]<end){end=roots[step];restart=step;}
                if(!(end>current))return 2;
                std::vector<Real> la(t,0),lb(t,0),cuts;
                cuts.reserve(t*t+2);cuts.push_back(current);cuts.push_back(end);
                for(int j=0;j<t;++j)for(int k=0;k<h;++k){la[j]+=(Real)wo[j*h+k]*ha[t*h+k];lb[j]+=(Real)wo[j*h+k]*hb[t*h+k];}
                for(int j=0;j<t;++j)if(!std::isfinite(la[j])||!std::isfinite(lb[j]))return 3;
                for(Real root:input_roots)if(root>current&&root<end){cuts.push_back(root);++count[3];}
                for(int i=0;i<t;++i)for(int j=i+1;j<t;++j) {
                    Real a=la[i]-la[j],b=lb[i]-lb[j];++count[2];
                    if(!std::isfinite(a)||!std::isfinite(b))return 3;
                    if(b!=0){Real root=-a/b;if(!std::isfinite(root))return 3;
                        if(root>current&&root<end){cuts.push_back(root);++count[3];}}
                }
                std::sort(cuts.begin(),cuts.end(),[&](Real a,Real b){++count[5];return a<b;});
                cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
                for(size_t j=1;j<cuts.size();++j) {
                    Real lo=cuts[j-1],hi=cuts[j];if(!(hi>lo))continue;
                    int limit=0;
                    // Root-side comparisons avoid forming a possibly rounded midpoint.
                    Real point=lo;
                    int winner=0,largest=0,second=-1;bool invalid=false;
                    for(int k=1;k<t;++k)if(compare(la[k],lb[k],la[winner],lb[winner],point,limit,invalid)>0)winner=k;
                    // Stable input sort: value first, index second; second largest is explicit.
                    for(int k=1;k<t;++k) {
                        int cmp=compare(input_a[k],input_b[k],input_a[largest],input_b[largest],point,limit,invalid);
                        if(cmp>=0){second=largest;largest=k;}
                        else if(second<0||compare(input_a[k],input_b[k],input_a[second],input_b[second],point,limit,invalid)>=0)second=k;
                    }
                    if(invalid)return 3;
                    Real angle=angle_width(lo,hi);count[6]+=(std::isfinite(lo)&&std::isfinite(hi)&&lo<0&&hi>0)?2:1;++count[4];full+=angle;
                    if(winner==second)mass+=angle;
                }
                current=end;
            }
        }
        answer[sample]=(double)(mass/(2*pi));coverage[sample]=(double)(full/(2*pi));
    }
    return 0;
}
