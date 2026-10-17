#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int64_t I;
typedef void (*gesdd_fn)(char*,I*,I*,double*,I*,double*,double*,I*,double*,I*,double*,I*,I*,I*);
static I stats[12];
void audit_reset(void){memset(stats,0,sizeof(stats));}
void audit_stats(I *out){memcpy(out,stats,sizeof(stats));}
void scipy_dgesdd_64_(char *job,I *m,I *n,double *a,I *lda,double *s,double *u,I *ldu,double *vt,I *ldvt,double *work,I *lwork,I *iwork,I *info){
 static gesdd_fn real=NULL;
 if(!real){void *h=dlopen("/workspace/clean-neurons/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f48b354e.so",RTLD_NOW|RTLD_LOCAL);if(!h){fputs(dlerror(),stderr);abort();}real=(gesdd_fn)dlsym(h,"scipy_dgesdd_64_");if(!real){fputs(dlerror(),stderr);abort();}}
 stats[0]++;stats[1]+=(*job=='N');stats[2]+=(*lwork==-1);stats[3]+=(*lwork!=-1);
 stats[4]=*m;stats[5]=*n;stats[6]=*lda;
 if(*lwork!=-1)stats[7]=*lwork;
 real(job,m,n,a,lda,s,u,ldu,vt,ldvt,work,lwork,iwork,info);
 stats[8]+=(*info==0);stats[9]+=(*info>0);stats[10]+=(*info<0);stats[11]=*info;
}

static I nested[10];
void nested_reset(void){memset(nested,0,sizeof(nested));}
void nested_stats(I *out){memcpy(out,nested,sizeof(nested));}
static void *resolve(const char *name){
 void *h=dlopen("/workspace/clean-neurons/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f48b354e.so",RTLD_NOW|RTLD_LOCAL);
 if(!h){fputs(dlerror(),stderr);abort();}void *f=dlsym(h,name);if(!f){fputs(dlerror(),stderr);abort();}return f;
}
void scipy_dlasq1_64_(I *n,double*d,double*e,double*w,I*info){
 typedef void(*fn)(I*,double*,double*,double*,I*);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dlasq1_64_");
 nested[0]++;nested[1]=*n;real(n,d,e,w,info);nested[2]+=(*info==0);nested[3]+=(*info!=0);
}
void scipy_dlasq2_64_(I *n,double*z,I*info){
 typedef void(*fn)(I*,double*,I*);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dlasq2_64_");
 nested[4]++;nested[5]=*n;real(n,z,info);nested[6]+=(*info==0);nested[7]+=(*info!=0);
 if(*info==0 && *n>2){nested[8]+=(I)z[2*(*n)+1];if((I)z[2*(*n)+1]>nested[9])nested[9]=(I)z[2*(*n)+1];}
}
