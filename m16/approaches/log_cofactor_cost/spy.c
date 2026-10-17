#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int64_t I;
typedef void (*getrf_fn)(I*,I*,double*,I*,I*,I*);
static I stats[14];
static int enabled=0;
static void *lookup(const char *lib,const char *name){void *h=dlopen(lib,RTLD_NOW|RTLD_LOCAL);if(!h){fputs(dlerror(),stderr);abort();}void *p=dlsym(h,name);if(!p){fputs(dlerror(),stderr);abort();}return p;}
void audit_start(void){memset(stats,0,sizeof(stats));enabled=1;}
void audit_stop(void){enabled=0;}
void audit_stats(I *out){memcpy(out,stats,sizeof(stats));}
void scipy_dgetrf_64_(I *m,I *n,double *a,I *lda,I *piv,I *info){
 static getrf_fn real=NULL;
 if(!real)real=(getrf_fn)lookup("/workspace/clean-neurons/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f48b354e.so","scipy_dgetrf_64_");
 if(enabled){stats[0]++;stats[1]+=(*m==3);stats[2]+=(*n==3);stats[3]+=(*lda==3);}
 real(m,n,a,lda,piv,info);
 if(enabled){stats[4]+=(*info==0);stats[5]+=(*info>0);stats[6]+=(*info<0);if(*m==3&&*n==3){for(I i=0;i<3;i++){stats[7]+=(piv[i]!=i+1);if(*info==0){stats[8]+=(a[i+i*(*lda)]<0);stats[9]+=!isfinite(a[i+i*(*lda)]);}}}}
}
double log(double x){static double(*real)(double)=NULL;if(!real)real=lookup("libm.so.6","log");if(enabled){stats[10]++;stats[11]+=!isfinite(x);}return real(x);}
double exp(double x){static double(*real)(double)=NULL;if(!real)real=lookup("libm.so.6","exp");if(enabled){stats[12]++;stats[13]+=!isfinite(x);}return real(x);}
