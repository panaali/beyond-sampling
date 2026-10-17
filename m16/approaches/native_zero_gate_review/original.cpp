#include "_source/dd_clip.cpp"
static constexpr int D=5;
#include <array>
#include <set>
#include <stdexcept>
extern "C" void dgesvd_(const char*,const char*,const int*,const int*,double*,const int*,double*,double*,const int*,double*,const int*,double*,const int*,int*);
struct Cell { Cone c;std::vector<double> A;std::array<uint64_t,2> all; };
struct Geometry {
 int h;const double *wi,*wh,*wo;const int* input_order;
 std::vector<double> good,bad,rays,maps;std::vector<uint64_t> bits;std::vector<int> headers;
 long long counts[D+8]={};
 Geometry(int H,const double* I,const double* W,const double* O,const int* order=nullptr):h(H),wi(I),wh(W),wo(O),input_order(order){}
 Result clip(const Cone& c,int d,const std::vector<double>& a,int cid,int want=3){Result r;int status=core(c.r.size()/d,d,c.r.data(),c.z.data(),a.data(),cid,want,r);++counts[D+5];if(status)throw status;return r;}
 int rank(const Cone& c,const std::vector<int>& indices){
  ++counts[D+4];int m=indices.size(),n=D,ld=std::max(1,m),one=1,info=0,lwork=std::max(5*std::min(m,n),3*std::min(m,n)+std::max(m,n));
  std::vector<double>a(m*n),s(std::min(m,n)),work(lwork);double unused=0;char job='N';
  for(int j=0;j<n;++j)for(int i=0;i<m;++i)a[j*m+i]=c.r[indices[i]*D+j];
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
 void record(const Cone& c,int type){headers.push_back(type);headers.push_back(c.r.size()/D);rays.insert(rays.end(),c.r.begin(),c.r.end());bits.insert(bits.end(),c.z.begin(),c.z.end());}
 void triangulate(const Cone& c,bool isgood){
  std::vector<int> F,used;uint64_t all[2]={0,0};for(size_t i=0;i<c.r.size()/D;++i){F.push_back(i);all[0]|=c.z[2*i];all[1]|=c.z[2*i+1];}
  for(int i=0;i<128;++i)if(all[i/64]&(uint64_t(1)<<(i%64)))used.push_back(i);
  std::vector<std::vector<int>> triangles;tri_rec(c,F,D,used,triangles);auto& dst=isgood?good:bad;
  for(const auto& tr:triangles){if(tr.size()!=D)throw 6;
   double A[D*D];for(int i=0;i<D;++i)for(int j=0;j<D;++j)A[i*D+j]=c.r[D*tr[j]+i];
   double det=1.;for(int j=0;j<D;++j){int pivot=j;double best=std::abs(A[j*D+j]);for(int i=j+1;i<D;++i)if(std::abs(A[i*D+j])>best){best=std::abs(A[i*D+j]);pivot=i;}
    if(best==0){det=0;break;}if(pivot!=j){for(int k=0;k<D;++k)std::swap(A[j*D+k],A[pivot*D+k]);det=-det;}
    for(int i=j+1;i<D;++i){double f=A[i*D+j]/A[j*D+j];for(int k=j+1;k<D;++k)A[i*D+k]-=f*A[j*D+k];}
   }if(det!=0)for(int j=0;j<D;++j)det*=A[j*D+j];
   if(det==0){++counts[D+7];continue;}
   for(int i:tr)for(int j=0;j<D;++j)dst.push_back(c.r[D*i+j]);++counts[isgood?D+2:D+3];
  }
 }
 void orthants(const Cone& original){
  std::vector<Cone> pieces={original};
  for(int k=0;k<D;++k){std::vector<Cone> next;std::vector<double>a(D,0);a[k]=1;
   for(const auto& c:pieces){Result r=clip(c,D,a,90+k);for(int side=0;side<2;++side)if(r.side[side].present&&r.side[side].r.size()/D>=D)next.push_back(std::move(r.side[side]));}
   pieces=std::move(next);
  }
  counts[D+6]+=pieces.size();
  for(auto& c:pieces){for(size_t i=0;i<c.r.size()/D;++i){double norm=0;for(int j=0;j<D;++j)norm+=std::abs(c.r[D*i+j]);if(!(norm>0)||!std::isfinite(norm))throw 2;for(int j=0;j<D;++j)c.r[D*i+j]/=norm;}
   record(c,3);triangulate(c,true);
  }
 }
 void finish(const Cell& cell){
  record(cell.c,0);maps.insert(maps.end(),cell.A.begin(),cell.A.end());
  double L[D*D]={0};bool dead=true;for(double x:cell.A)if(x!=0)dead=false;
  for(int j=0;j<D;++j)for(int q=0;q<D;++q)for(int k=0;k<h;++k)L[j*D+q]+=wo[j*h+k]*cell.A[k*D+q];
  std::vector<Cone> goodpieces,badpieces;
  for(int j=0;j<D;++j){if(dead&&j!=0)continue;Cone Q=cell.c;
   if(!dead)for(int k=0;k<D;++k)if(k!=j){std::vector<double> dl(D);double maxabs=0;for(int q=0;q<D;++q){dl[q]=L[D*j+q]-L[D*k+q];maxabs=std::max(maxabs,std::abs(dl[q]));}
    if(maxabs==0){if(k<j){Q.present=false;break;}continue;}
    Q=clip(Q,D,dl,60+k,1).side[0];if(!Q.present)break;
   }
   if(!Q.present)continue;std::vector<std::pair<Cone,int>> S={{Q,0}};
   for(int k=0;k<D;++k)if(k!=j){std::vector<double>a(D,0);a[j]=1;a[k]=-1;std::vector<std::pair<Cone,int>> next;
    for(auto& pair:S){Result r=clip(pair.first,D,a,60+D+k);if(r.side[0].present)next.push_back({r.side[0],pair.second});
     if(r.side[1].present){if(pair.second+1>=2)badpieces.push_back(r.side[1]);else next.push_back({r.side[1],pair.second+1});}}
    S=std::move(next);if(S.empty())break;
   }
   for(auto& pair:S)(pair.second==1?goodpieces:badpieces).push_back(pair.first);
  }
  counts[D]+=goodpieces.size();counts[D+1]+=badpieces.size();
  for(auto& c:goodpieces){record(c,1);orthants(c);}for(auto& c:badpieces)record(c,2);
 }
 void enumerate(const Cell& cell,int t){
  if(t==D){finish(cell);return;}
  std::vector<int> order(input_order,input_order+h);int k0=order[0];if(wi[k0]==0)throw 1;
  std::vector<double>B(h*t,0);for(int j=0;j<h;++j)for(int q=0;q<t;++q)for(int k=0;k<h;++k)B[j*t+q]+=wh[j*h+k]*cell.A[k*D+q];
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
  for(auto& p:pieces){Cell child;child.c=std::move(p.c);child.all=all;child.A=cell.A;for(int j=0;j<h;++j){for(int q=0;q<t;++q)child.A[D*j+q]=B[j*t+q];child.A[D*j+t]=wi[j];if(!p.gates[j])for(int q=0;q<D;++q)child.A[D*j+q]=0;}enumerate(child,t+1);}
 }
 void run(){counts[0]=2;for(double sign:{1.,-1.}){Cell c;c.c.present=true;c.c.r={sign};c.c.z={0,0};c.A.assign(h*D,0);c.all={1,0};for(int j=0;j<h;++j)if(sign*wi[j]>0)c.A[D*j]=wi[j];enumerate(c,1);}}
};
extern "C" void* dd_geometry(int h,const double* wi,const double* wh,const double* wo,const int* order,int* status){
 if(h!=2){*status=4;return nullptr;}Geometry* g=new Geometry(h,wi,wh,wo,order);try{g->run();*status=0;return g;}catch(int code){*status=code;delete g;return nullptr;}catch(...){*status=9;delete g;return nullptr;}}
extern "C" const void* dd_get(void* ptr,int tag,long long* size){Geometry& g=*static_cast<Geometry*>(ptr);switch(tag){
 case 0:*size=g.good.size();return g.good.data();case 1:*size=g.bad.size();return g.bad.data();case 2:*size=g.rays.size();return g.rays.data();case 3:*size=g.bits.size();return g.bits.data();case 4:*size=g.headers.size();return g.headers.data();case 5:*size=g.maps.size();return g.maps.data();default:*size=D+8;return g.counts;}}
extern "C" void dd_free(void* ptr){delete static_cast<Geometry*>(ptr);}
extern "C" int dd_rank_test(int m,const double* matrix){Geometry g(1,nullptr,nullptr,nullptr);Cone c;c.r.assign(matrix,matrix+D*m);std::vector<int>indices(m);for(int i=0;i<m;++i)indices[i]=i;try{return g.rank(c,indices);}catch(...){return -1;}}
