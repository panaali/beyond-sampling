#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int64_t I;
static I events[64][13],count;
static int enabled;
static void *lookup(const char *name){void *h=dlopen("/workspace/clean-neurons/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f48b354e.so",RTLD_NOW|RTLD_LOCAL);if(!h){fputs(dlerror(),stderr);abort();}void *p=dlsym(h,name);if(!p){fputs(dlerror(),stderr);abort();}return p;}
static I bits(double x){I r;memcpy(&r,&x,8);return r;}
static void record(I kind,I order,I t1,I t2,I m,I n,I k,I lda,I ldb,I ldc,double alpha,double beta,I same){if(!enabled)return;if(count<64){I row[13]={kind,order,t1,t2,m,n,k,lda,ldb,ldc,bits(alpha),bits(beta),same};memcpy(events[count],row,sizeof(row));}count++;}
void audit_start(void){memset(events,0,sizeof(events));count=0;enabled=1;}
void audit_stop(void){enabled=0;}
I audit_events(I*out){if(count>64)return -count;memcpy(out,events,sizeof(events));return count;}
void scipy_cblas_dgemm64_(int order,int ta,int tb,I m,I n,I k,double alpha,const double*A,I lda,const double*B,I ldb,double beta,double*C,I ldc){
 typedef void(*fn)(int,int,int,I,I,I,double,const double*,I,const double*,I,double,double*,I);static fn real;
 if(!real)real=(fn)lookup("scipy_cblas_dgemm64_");record(0,order,ta,tb,m,n,k,lda,ldb,ldc,alpha,beta,A==B);real(order,ta,tb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}
void scipy_cblas_dsyrk64_(int order,int uplo,int trans,I n,I k,double alpha,const double*A,I lda,double beta,double*C,I ldc){
 typedef void(*fn)(int,int,int,I,I,double,const double*,I,double,double*,I);static fn real;
 if(!real)real=(fn)lookup("scipy_cblas_dsyrk64_");record(1,order,uplo,trans,n,n,k,lda,0,ldc,alpha,beta,1);real(order,uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
}
