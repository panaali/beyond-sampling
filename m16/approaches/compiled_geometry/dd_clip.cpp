#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
struct Cone { std::vector<double> r; std::vector<uint64_t> z; bool present=false; };
struct Result { Cone side[2]; long long clip=0,adj=0; };
static int core(int k,int d,const double* R,const uint64_t* Z,const double* a,int cid,int want,Result& out) {
    if(k<1||d<1||cid<0||cid>=128)return 4;
    double norm=0;for(int j=0;j<d;++j)norm+=a[j]*a[j];norm=std::sqrt(norm);
    if(!(norm>0))return 1;if(!std::isfinite(norm))return 2;
    std::vector<double> unit(d),v(k);for(int j=0;j<d;++j)unit[j]=a[j]/norm;
    std::vector<int> p,m,z;
    for(int i=0;i<k;++i){double x=0;for(int j=0;j<d;++j)x+=R[i*d+j]*unit[j];if(!std::isfinite(x))return 2;v[i]=x;
        if(x>1e-11)p.push_back(i);else if(x< -1e-11)m.push_back(i);else z.push_back(i);}
    out.clip=3*d+20+2*k*d;
    uint64_t bit[2]={0,0};bit[cid/64]=uint64_t(1)<<(cid%64);
    auto append=[&](Cone& c,int i,bool tight){for(int j=0;j<d;++j)c.r.push_back(R[i*d+j]);for(int q=0;q<2;++q)c.z.push_back(Z[2*i+q]|(tight?bit[q]:0));};
    if(p.empty()||m.empty()){
        int side=m.empty()?0:1;
        if(want&(1<<side)){out.side[side].present=true;for(int i=0;i<k;++i)append(out.side[side],i,std::abs(v[i])<=1e-11);}
        return 0;
    }
    std::vector<double> nr;std::vector<uint64_t> nz;
    out.adj+=(long long)p.size()*m.size()*4;
    for(int pi:p)for(int mi:m){uint64_t common[2]={Z[2*pi]&Z[2*mi],Z[2*pi+1]&Z[2*mi+1]};
        if(__builtin_popcountll(common[0])+__builtin_popcountll(common[1])<d-2)continue;
        out.adj+=(long long)k*4;int contained=0;
        for(int i=0;i<k;++i)if((common[0]&~Z[2*i])==0&&(common[1]&~Z[2*i+1])==0)++contained;
        if(contained!=2)continue;
        std::vector<double> ray(d);double sq=0;
        for(int j=0;j<d;++j){ray[j]=v[pi]*R[mi*d+j]-v[mi]*R[pi*d+j];sq+=ray[j]*ray[j];}
        double length=std::sqrt(sq);if(!(length>0)||!std::isfinite(length))return 2;
        for(double x:ray)nr.push_back(x/length);nz.push_back(common[0]|bit[0]);nz.push_back(common[1]|bit[1]);out.clip+=7*d+20;
    }
    for(int side=0;side<2;++side)if(want&(1<<side)){
        Cone& c=out.side[side];c.present=true;for(int i:side?m:p)append(c,i,false);for(int i:z)append(c,i,true);
        c.r.insert(c.r.end(),nr.begin(),nr.end());c.z.insert(c.z.end(),nz.begin(),nz.end());}
    return 0;
}
extern "C" int dd_clip(int k,int d,const double* R,const uint64_t* Z,const double* a,int cid,int want,int capacity,double* outR,uint64_t* outZ,int* sizes,long long* ops){
    Result out;int status=core(k,d,R,Z,a,cid,want,out);if(status)return status;
    ops[0]=out.clip;ops[1]=out.adj;
    for(int s=0;s<2;++s){sizes[s]=out.side[s].present?out.side[s].r.size()/d:-1;if(sizes[s]>capacity)return 3;
        std::copy(out.side[s].r.begin(),out.side[s].r.end(),outR+s*capacity*d);
        std::copy(out.side[s].z.begin(),out.side[s].z.end(),outZ+s*capacity*2);}
    return 0;
}
extern "C" int dd_replay(int n,const int* meta,const long long* offsets,const double* R,const uint64_t* Z,const double* a,double* checksum,long long* totals){
    totals[0]=totals[1]=0;
    for(int c=0;c<n;++c){Result out;int status=core(meta[4*c],meta[4*c+1],R+offsets[3*c],Z+offsets[3*c+1],a+offsets[3*c+2],meta[4*c+2],meta[4*c+3],out);if(status)return status;
        double sum=0;for(int s=0;s<2;++s)for(double x:out.side[s].r)sum+=x;checksum[c]=sum;totals[0]+=out.clip;totals[1]+=out.adj;}
    return 0;
}
