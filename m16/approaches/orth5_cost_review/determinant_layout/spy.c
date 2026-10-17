#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int64_t I;
typedef void (*getrf_fn)(I*,I*,double*,I*,I*,I*);
typedef struct { I m,n,lda,info; I piv[4]; double before[16]; } Event;
static Event events[256];
static I used=0,overflow=0,unsupported=0;
static int enabled=0;
static getrf_fn real_fn=NULL;
static getrf_fn resolve(void){
    if(!real_fn){
        void *h=dlopen("/workspace/clean-neurons/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f48b354e.so",RTLD_NOW|RTLD_LOCAL);
        if(!h){fputs(dlerror(),stderr);abort();}
        real_fn=(getrf_fn)dlsym(h,"scipy_dgetrf_64_");
        if(!real_fn){fputs(dlerror(),stderr);abort();}
    }
    return real_fn;
}
uintptr_t audit_real_address(void){return (uintptr_t)resolve();}
void audit_start(void){memset(events,0,sizeof(events));used=overflow=unsupported=0;enabled=1;}
void audit_stop(void){enabled=0;}
I audit_copy(Event *out,I capacity){
    if(overflow||unsupported||capacity<used)return -1;
    memcpy(out,events,(size_t)used*sizeof(Event));return used;
}
void scipy_dgetrf_64_(I *m,I *n,double *a,I *lda,I *piv,I *info){
    getrf_fn fn=resolve();Event *e=NULL;
    if(enabled){
        if(used>=256)overflow=1;
        else{
            e=&events[used++];e->m=*m;e->n=*n;e->lda=*lda;
            if(*m!=*n||(*n!=3&&*n!=4)||*lda<*m){unsupported=1;e=NULL;}
            else for(I i=0;i<*m;i++)for(I j=0;j<*n;j++)e->before[4*i+j]=a[i+j*(*lda)];
        }
    }
    fn(m,n,a,lda,piv,info);
    if(e){e->info=*info;for(I i=0;i<*n;i++)e->piv[i]=piv[i];}
}
