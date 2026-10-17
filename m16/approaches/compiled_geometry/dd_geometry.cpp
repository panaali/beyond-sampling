#include "dd_clip.cpp"
#include <array>
#include <set>
#include <stdexcept>
extern "C" void dgesvd_(const char*,const char*,const int*,const int*,double*,const int*,double*,double*,const int*,double*,const int*,double*,const int*,int*);
struct Cell { Cone c;std::vector<double> A;std::array<uint64_t,2> all; };
struct Geometry {
 int h;const double *wi,*wh,*wo;const int* input_order;
 std::vector<double> good,bad,rays,maps;std::vector<uint64_t> bits;std::vector<int> headers;
 long long counts[9]={0,0,0,0,0,0,0,0,0};
 Geometry(int H,const double* I,const double* W,const double* O,const int* order=nullptr):h(H),wi(I),wh(W),wo(O),input_order(order){}
 Result clip(const Cone& c,int d,const std::vector<double>& a,int cid,int want=3){Result r;int status=core(c.r.size()/d,d,c.r.data(),c.z.data(),a.data(),cid,want,r);++counts[8];if(status)throw status;return r;}
 int rank(const Cone& c,const std::vector<int>& indices){
  ++counts[7];int m=indices.size(),n=3,ld=std::max(1,m),one=1,info=0,lwork=std::max(5*std::min(m,n),3*std::min(m,n)+std::max(m,n));
  std::vector<double>a(m*n),s(std::min(m,n)),work(lwork);double unused=0;char job='N';
  for(int j=0;j<n;++j)for(int i=0;i<m;++i)a[j*m+i]=c.r[indices[i]*3+j];
  dgesvd_(&job,&job,&m,&n,a.data(),&ld,s.data(),&unused,&one,&unused,&one,work.data(),&lwork,&info);if(info)throw 5;
  int r=0;for(double x:s)if(x>1e-9)++r;return r;
 }
 void tri_rec(const Cone& c,const std::vector<int>& F,int f,const std::vector<int>& used,std::vector<std::vector<int>>& out){
  if((int)F.size()==f){out.push_back(F);return;}
  int v=F[0];std::set<std::vector<int>> seen;
  for(int cid:used){uint64_t b=uint64_t(1)<<(cid%64);if(c.z[2*v+cid/64]&b)continue;
   std::vector<int> G;for(int i:F)if(c.z[2*i+cid/64]&b)G.push_back(i);
   if(G.empty()||G.size()==F.size()||(int)G.size()<f-1||!seen.insert(G).second)continue;
   if(rank(c,G)!=f-1)continue;size_t n0=out.size();tri_rec(c,G,f-1,used,out);for(size_t j=n0;j<out.size();++j)out[j].insert(out[j].begin(),v);
  }
 }
 void record(const Cone& c,int type){headers.push_back(type);headers.push_back(c.r.size()/3);rays.insert(rays.end(),c.r.begin(),c.r.end());bits.insert(bits.end(),c.z.begin(),c.z.end());}
 void triangulate(const Cone& c,bool isgood){
  std::vector<int> F,used;uint64_t all[2]={0,0};for(size_t i=0;i<c.r.size()/3;++i){F.push_back(i);all[0]|=c.z[2*i];all[1]|=c.z[2*i+1];}
  for(int i=0;i<128;++i)if(all[i/64]&(uint64_t(1)<<(i%64)))used.push_back(i);
  std::vector<std::vector<int>> triangles;tri_rec(c,F,3,used,triangles);auto& dst=isgood?good:bad;
  for(const auto& tr:triangles){if(tr.size()!=3)throw 6;for(int i:tr)for(int j=0;j<3;++j)dst.push_back(c.r[3*i+j]);}
  counts[isgood?5:6]+=triangles.size();
 }
 void finish(const Cell& cell){
  record(cell.c,0);maps.insert(maps.end(),cell.A.begin(),cell.A.end());
  double L[9]={0};bool dead=true;for(double x:cell.A)if(x!=0)dead=false;
  for(int j=0;j<3;++j)for(int q=0;q<3;++q)for(int k=0;k<h;++k)L[j*3+q]+=wo[j*h+k]*cell.A[k*3+q];
  double scale=0;for(double x:L)scale=std::max(scale,std::abs(x));std::vector<Cone> goodpieces,badpieces;
  for(int j=0;j<3;++j){if(dead&&j!=0)continue;Cone Q=cell.c;
   if(!dead)for(int k=0;k<3;++k)if(k!=j){std::vector<double> dl(3);double maxabs=0;for(int q=0;q<3;++q){dl[q]=L[3*j+q]-L[3*k+q];maxabs=std::max(maxabs,std::abs(dl[q]));}
    if(maxabs<=1e-13*scale){if(k<j){Q.present=false;break;}continue;}
    Q=clip(Q,3,dl,33+k,1).side[0];if(!Q.present)break;
   }
   if(!Q.present)continue;std::vector<std::pair<Cone,int>> S={{Q,0}};
   for(int k=0;k<3;++k)if(k!=j){std::vector<double>a(3,0);a[j]=1;a[k]=-1;std::vector<std::pair<Cone,int>> next;
    for(auto& pair:S){Result r=clip(pair.first,3,a,36+k);if(r.side[0].present)next.push_back({r.side[0],pair.second});
     if(r.side[1].present){if(pair.second+1>=2)badpieces.push_back(r.side[1]);else next.push_back({r.side[1],pair.second+1});}}
    S=std::move(next);if(S.empty())break;
   }
   for(auto& pair:S)(pair.second==1?goodpieces:badpieces).push_back(pair.first);
  }
  counts[3]+=goodpieces.size();counts[4]+=badpieces.size();
  for(auto& c:goodpieces){record(c,1);triangulate(c,true);}for(auto& c:badpieces){record(c,2);triangulate(c,false);}
 }
 void enumerate(const Cell& cell,int t){
  if(t==3){finish(cell);return;}
  std::vector<int> order(input_order,input_order+h);int k0=order[0];if(wi[k0]==0)throw 1;
  std::vector<double>B(h*t,0);for(int j=0;j<h;++j)for(int q=0;q<t;++q)for(int k=0;k<h;++k)B[j*t+q]+=wh[j*h+k]*cell.A[k*3+q];
  struct Piece{Cone c;std::vector<bool> gates;};std::vector<Piece> pieces;
  for(double sign:{1.,-1.}){Piece p;p.c.present=true;p.gates.assign(h,false);p.gates[k0]=sign>0;int cid=1+16*(t-1)+k0;
   for(size_t i=0;i<cell.c.r.size()/t;++i){std::vector<double> ray(t+1);double y=0,sq=0;for(int q=0;q<t;++q){ray[q]=cell.c.r[i*t+q];y+=ray[q]*sign*B[k0*t+q];}ray[t]=-y/(sign*wi[k0]);for(double x:ray)sq+=x*x;double norm=std::sqrt(sq);if(!(norm>0)||!std::isfinite(norm))throw 2;for(double x:ray)p.c.r.push_back(x/norm);
    for(int q=0;q<2;++q)p.c.z.push_back(cell.c.z[2*i+q]|(q==cid/64?(uint64_t(1)<<(cid%64)):0));}
   for(int q=0;q<=t;++q)p.c.r.push_back(q==t?(sign*wi[k0]>0?1.:-1.):0.);for(int q=0;q<2;++q)p.c.z.push_back(cell.all[q]);pieces.push_back(std::move(p));
  }
  for(int oi=1;oi<h;++oi){int k=order[oi],cid=1+16*(t-1)+k;std::vector<double>a(t+1);for(int q=0;q<t;++q)a[q]=B[k*t+q];a[t]=wi[k];std::vector<Piece> next;
   for(auto& p:pieces){Result r=clip(p.c,t+1,a,cid);if(r.side[0].present){auto g=p.gates;g[k]=true;next.push_back({std::move(r.side[0]),std::move(g)});}if(r.side[1].present)next.push_back({std::move(r.side[1]),p.gates});}pieces=std::move(next);
  }
  auto all=cell.all;for(int k=0;k<h;++k){int cid=1+16*(t-1)+k;all[cid/64]|=uint64_t(1)<<(cid%64);}counts[t]+=pieces.size();
  for(auto& p:pieces){Cell child;child.c=std::move(p.c);child.all=all;child.A=cell.A;for(int j=0;j<h;++j){for(int q=0;q<t;++q)child.A[3*j+q]=B[j*t+q];child.A[3*j+t]=wi[j];if(!p.gates[j])for(int q=0;q<3;++q)child.A[3*j+q]=0;}enumerate(child,t+1);}
 }
 void run(){counts[0]=2;for(double sign:{1.,-1.}){Cell c;c.c.present=true;c.c.r={sign};c.c.z={0,0};c.A.assign(h*3,0);c.all={1,0};for(int j=0;j<h;++j)if(sign*wi[j]>0)c.A[3*j]=wi[j];enumerate(c,1);}}
};
extern "C" void* dd_geometry(int h,const double* wi,const double* wh,const double* wo,const int* order,int* status){
 if(h<1||h>16){*status=4;return nullptr;}Geometry* g=new Geometry(h,wi,wh,wo,order);try{g->run();*status=0;return g;}catch(int code){*status=code;delete g;return nullptr;}catch(...){*status=9;delete g;return nullptr;}}
extern "C" const void* dd_get(void* ptr,int tag,long long* size){Geometry& g=*static_cast<Geometry*>(ptr);switch(tag){
 case 0:*size=g.good.size();return g.good.data();case 1:*size=g.bad.size();return g.bad.data();case 2:*size=g.rays.size();return g.rays.data();case 3:*size=g.bits.size();return g.bits.data();case 4:*size=g.headers.size();return g.headers.data();case 5:*size=g.maps.size();return g.maps.data();default:*size=9;return g.counts;}}
extern "C" void dd_free(void* ptr){delete static_cast<Geometry*>(ptr);}
extern "C" int dd_rank_test(int m,const double* matrix){Geometry g(1,nullptr,nullptr,nullptr);Cone c;c.r.assign(matrix,matrix+3*m);std::vector<int>indices(m);for(int i=0;i<m;++i)indices[i]=i;try{return g.rank(c,indices);}catch(...){return -1;}}
