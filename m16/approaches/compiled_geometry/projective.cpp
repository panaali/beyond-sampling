#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using Real=long double;
static const Real INF=std::numeric_limits<Real>::infinity();

// Compare affine functions at a finite point or the limiting infinite endpoint.
static int compare(Real a,Real b,Real c,Real d,Real point,int limit,bool& invalid) {
    if(limit) {
        int order=((b>d)-(b<d))*limit;
        if(order)return order;
        return (a>c)-(a<c);
    }
    Real lhs=a+b*point,rhs=c+d*point;
    if(!std::isfinite(lhs)||!std::isfinite(rhs)){invalid=true;return 0;}
    return (lhs>rhs)-(lhs<rhs);
}

extern "C" int projective_walk(int n,int h,int t,const double* wi,const double* wh,
    const double* wo,const double* U,const double* V,int cap,double* answer,
    double* coverage,long long* counters) {
    const Real pi=acosl(-1.L);
    for(int sample=0;sample<n;++sample) {
        long long* count=counters+sample*7;
        for(int j=0;j<7;++j)count[j]=0;
        Real mass=0,full=0;
        for(int sign:{1,-1}) {
            std::vector<Real> input_a(t),input_b(t),input_roots;
            for(int j=0;j<t;++j){input_a[j]=sign*(Real)U[sample*t+j];input_b[j]=sign*(Real)V[sample*t+j];}
            for(int i=0;i<t;++i)for(int j=i+1;j<t;++j) {
                Real a=input_a[i]-input_a[j],b=input_b[i]-input_b[j];++count[2];
                if(b!=0){Real root=-a/b;if(!std::isfinite(root))return 3;input_roots.push_back(root);}
            }
            Real current=-INF;
            while(current<INF) {
                if(count[0]>=cap)return 1;
                ++count[0];
                std::vector<Real> ha(h,0),hb(h,0),nexta(h),nextb(h);
                Real end=INF;
                for(int step=0;step<t;++step) {
                    ++count[1];
                    for(int j=0;j<h;++j) {
                        Real a=(Real)wi[j]*input_a[step],b=(Real)wi[j]*input_b[step];
                        for(int k=0;k<h;++k){a+=(Real)wh[j*h+k]*ha[k];b+=(Real)wh[j*h+k]*hb[k];}
                        if(!std::isfinite(a)||!std::isfinite(b))return 3;
                        bool active=a>0;
                        ++count[2];
                        if(b!=0) {
                            Real root=-a/b;if(!std::isfinite(root))return 3;
                            active=b>0?current>=root:current<root;
                            if(root>current&&root<end)end=root;
                        }
                        nexta[j]=active?a:0;nextb[j]=active?b:0;
                    }
                    ha.swap(nexta);hb.swap(nextb);
                }
                if(!(end>current))return 2;
                std::vector<Real> la(t,0),lb(t,0),cuts;
                cuts.reserve(t*t+2);cuts.push_back(current);cuts.push_back(end);
                for(int j=0;j<t;++j)for(int k=0;k<h;++k){la[j]+=(Real)wo[j*h+k]*ha[k];lb[j]+=(Real)wo[j*h+k]*hb[k];}
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
                    int limit=std::isinf(lo)?-1:(std::isinf(hi)?1:0);
                    // If the entire real line has no pair root, either limit has the same ordering.
                    Real point=limit?0:.5L*lo+.5L*hi;
                    int winner=0,largest=0,second=-1;bool invalid=false;
                    for(int k=1;k<t;++k)if(compare(la[k],lb[k],la[winner],lb[winner],point,limit,invalid)>0)winner=k;
                    // Stable input sort: value first, index second; second largest is explicit.
                    for(int k=1;k<t;++k) {
                        int cmp=compare(input_a[k],input_b[k],input_a[largest],input_b[largest],point,limit,invalid);
                        if(cmp>=0){second=largest;largest=k;}
                        else if(second<0||compare(input_a[k],input_b[k],input_a[second],input_b[second],point,limit,invalid)>=0)second=k;
                    }
                    if(invalid)return 3;
                    Real angle=atanl(hi)-atanl(lo);count[6]+=2;++count[4];full+=angle;
                    if(winner==second)mass+=angle;
                }
                current=end;
            }
        }
        answer[sample]=(double)(mass/(2*pi));coverage[sample]=(double)(full/(2*pi));
    }
    return 0;
}
