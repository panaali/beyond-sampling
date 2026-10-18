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
