#include <stdint.h>
#include <string.h>
typedef int64_t I;
#define CAP 4096
#define N 16
static I events[CAP][10],count;
static uintptr_t table;
static void *original[N];
static const I offsets[N]={0x390,0x3c8,0x3d0,0x3d8,0x3e0,0x3e8,0x3f0,0x3f8,0x400,0x408,0x410,0x418,0x420,0x428,0x430,0x438};
static I bits(double x){I r;memcpy(&r,&x,8);return r;}
static void record(I kind,I m,I n,I k,I lda,I ldb,I ldc,double alpha,double beta,I result){if(count<CAP){I row[10]={kind,m,n,k,lda,ldb,ldc,bits(alpha),bits(beta),result};memcpy(events[count],row,sizeof(row));}count++;}
static int scal(I n,I d0,I d1,double alpha,double*x,I ix,double*y,I iy,double*z,I iz){record(0,n,0,0,ix,iy,iz,alpha,0,0);return ((int(*)(I,I,I,double,double*,I,double*,I,double*,I))original[0])(n,d0,d1,alpha,x,ix,y,iy,z,iz);}
static int gemm(I m,I n,I k,double alpha,double*A,double*B,double*C,I ldc){record(1,m,n,k,0,0,ldc,alpha,0,0);return ((int(*)(I,I,I,double,double*,double*,double*,I))original[1])(m,n,k,alpha,A,B,C,ldc);}
static int beta(I m,I n,I d0,double v,double*A,I lda,double*B,I ldb,double*C,I ldc){record(2,m,n,d0,lda,ldb,ldc,0,v,0);return ((int(*)(I,I,I,double,double*,I,double*,I,double*,I))original[2])(m,n,d0,v,A,lda,B,ldb,C,ldc);}
#define COPY(NAME,IDX) static int NAME(I m,I n,double*A,I lda,double*B){record(IDX,m,n,0,lda,0,0,0,0,0);return ((int(*)(I,I,double*,I,double*))original[IDX])(m,n,A,lda,B);}
COPY(incopy,3) COPY(itcopy,4) COPY(oncopy,5) COPY(otcopy,6)
static int permit(int ta,int tb,I m,I n,I k,double alpha,double beta){int r=((int(*)(int,int,I,I,I,double,double))original[7])(ta,tb,m,n,k,alpha,beta);record(7,m,n,k,ta,tb,0,alpha,beta,r);return r;}
#define SMALL(NAME,IDX) static int NAME(I m,I n,I k,double*A,I lda,double alpha,double*B,I ldb,double beta,double*C,I ldc){record(IDX,m,n,k,lda,ldb,ldc,alpha,beta,0);return ((int(*)(I,I,I,double*,I,double,double*,I,double,double*,I))original[IDX])(m,n,k,A,lda,alpha,B,ldb,beta,C,ldc);}
SMALL(nn,8) SMALL(nt,9) SMALL(tn,10) SMALL(tt,11)
#define B0(NAME,IDX) static int NAME(I m,I n,I k,double*A,I lda,double alpha,double*B,I ldb,double*C,I ldc){record(IDX,m,n,k,lda,ldb,ldc,alpha,0,0);return ((int(*)(I,I,I,double*,I,double,double*,I,double*,I))original[IDX])(m,n,k,A,lda,alpha,B,ldb,C,ldc);}
B0(nn0,12) B0(nt0,13) B0(tn0,14) B0(tt0,15)
void kernel_install(uintptr_t address){void*wrap[N]={(void*)scal,(void*)gemm,(void*)beta,(void*)incopy,(void*)itcopy,(void*)oncopy,(void*)otcopy,(void*)permit,(void*)nn,(void*)nt,(void*)tn,(void*)tt,(void*)nn0,(void*)nt0,(void*)tn0,(void*)tt0};table=address;for(int i=0;i<N;i++){void**p=(void**)(table+offsets[i]);original[i]=*p;*p=wrap[i];}}
void kernel_restore(void){for(int i=0;i<N;i++)*(void**)(table+offsets[i])=original[i];}
void kernel_reset(void){count=0;memset(events,0,sizeof(events));}
I kernel_events(I*out){if(count>CAP)return -count;memcpy(out,events,sizeof(events));return count;}
