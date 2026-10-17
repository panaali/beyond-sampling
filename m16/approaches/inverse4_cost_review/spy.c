#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int64_t I;
typedef void (*gesv_fn)(I*,I*,double*,I*,I*,double*,I*,I*);
static I stats[12];
void audit_reset(void){memset(stats,0,sizeof(stats));}
void audit_stats(I *out){memcpy(out,stats,sizeof(stats));}
void scipy_dgesv_64_(I *n,I *nrhs,double *a,I *lda,I *ipiv,double *b,I *ldb,I *info){
 static gesv_fn real=NULL;
 if(!real){void *handle=dlopen("/workspace/clean-neurons/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f48b354e.so",RTLD_NOW|RTLD_LOCAL);if(!handle){fputs(dlerror(),stderr);abort();}real=(gesv_fn)dlsym(handle,"scipy_dgesv_64_");if(!real){fputs(dlerror(),stderr);abort();}}
 stats[0]++;stats[1]+=(*n==4);stats[2]+=(*nrhs==4);stats[3]+=(*lda==4);stats[4]+=(*ldb==4);
 I identity=1,finite=1;
 if(*n==4 && *nrhs==4){for(I j=0;j<4;j++)for(I i=0;i<4;i++){identity&=(b[i+j*(*ldb)]==(i==j?1.:0.));finite&=isfinite(a[i+j*(*lda)]);}}else identity=finite=0;
 stats[5]+=identity;stats[6]+=finite;
 real(n,nrhs,a,lda,ipiv,b,ldb,info);
 stats[7]+=(*info==0);stats[8]+=(*info>0);stats[9]+=(*info<0);
 if(*n==4){for(I i=0;i<4;i++)stats[10]+=(ipiv[i]!=i+1);}
 stats[11]=*info;
}
