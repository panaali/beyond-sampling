#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef int64_t I;
static I rows[512][5],used;
static void record(I id,I a,I b,I c,I d){if(used<512){rows[used][0]=id;rows[used][1]=a;rows[used][2]=b;rows[used][3]=c;rows[used][4]=d;}used++;}
void reset(void){used=0;memset(rows,0,sizeof(rows));}
I events(I*out){if(used>512)return -used;memcpy(out,rows,used*5*sizeof(I));return used;}
static void*resolve(const char*n){void*h=dlopen("/workspace/clean-neurons/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f48b354e.so",RTLD_NOW|RTLD_LOCAL);if(!h)abort();void*f=dlsym(h,n);if(!f)abort();return f;}
void scipy_dgesdd_64_(char *job,I *m,I *n,double *a,I *lda,double *s,double *u,I *ldu,double *vt,I *ldvt,double *work,I *lwork,I *iwork,I *info){typedef void(*fn)(char *,I *,I *,double *,I *,double *,double *,I *,double *,I *,double *,I *,I *,I *);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dgesdd_64_");real(job,m,n,a,lda,s,u,ldu,vt,ldvt,work,lwork,iwork,info);record(0,*m,*n,*lwork==-1,*job);}
void scipy_dsyevd_64_(char *job,char *uplo,I *n,double *a,I *lda,double *w,double *work,I *lwork,I *iwork,I *liwork,I *info){typedef void(*fn)(char *,char *,I *,double *,I *,double *,double *,I *,I *,I *,I *);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dsyevd_64_");real(job,uplo,n,a,lda,w,work,lwork,iwork,liwork,info);record(1,*n,*job,*lwork==-1,*uplo);}
void scipy_dgelqf_64_(I *m,I *n,double *a,I *lda,double *tau,double *work,I *lwork,I *info){typedef void(*fn)(I *,I *,double *,I *,double *,double *,I *,I *);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dgelqf_64_");real(m,n,a,lda,tau,work,lwork,info);record(2,*m,*n,*lwork==-1,*info);}
void scipy_dorglq_64_(I *m,I *n,I *k,double *a,I *lda,double *tau,double *work,I *lwork,I *info){typedef void(*fn)(I *,I *,I *,double *,I *,double *,double *,I *,I *);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dorglq_64_");real(m,n,k,a,lda,tau,work,lwork,info);record(3,*m,*n,*k,*lwork==-1);}
void scipy_dsytrd_64_(char *uplo,I *n,double *a,I *lda,double *d,double *e,double *tau,double *work,I *lwork,I *info){typedef void(*fn)(char *,I *,double *,I *,double *,double *,double *,double *,I *,I *);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dsytrd_64_");real(uplo,n,a,lda,d,e,tau,work,lwork,info);record(4,*n,*uplo,*lwork==-1,*info);}
void scipy_dsterf_64_(I *n,double *d,double *e,I *info){typedef void(*fn)(I *,double *,double *,I *);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dsterf_64_");real(n,d,e,info);record(5,*n,*info,0,0);}
void scipy_dlarfg_64_(I *n,double *alpha,double *x,I *incx,double *tau){typedef void(*fn)(I *,double *,double *,I *,double *);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dlarfg_64_");real(n,alpha,x,incx,tau);record(6,*n,*incx,0,0);}
void scipy_dsytd2_64_(char *uplo,I *n,double *a,I *lda,double *d,double *e,double *tau,I *info){typedef void(*fn)(char *,I *,double *,I *,double *,double *,double *,I *);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dsytd2_64_");real(uplo,n,a,lda,d,e,tau,info);record(7,*n,*uplo,*info,0);}
void scipy_dorgl2_64_(I *m,I *n,I *k,double *a,I *lda,double *tau,double *work,I *info){typedef void(*fn)(I *,I *,I *,double *,I *,double *,double *,I *);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dorgl2_64_");real(m,n,k,a,lda,tau,work,info);record(8,*m,*n,*k,*info);}
void scipy_dgebrd_64_(I *m,I *n,double *a,I *lda,double *d,double *e,double *tauq,double *taup,double *work,I *lwork,I *info){typedef void(*fn)(I *,I *,double *,I *,double *,double *,double *,double *,double *,I *,I *);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dgebrd_64_");real(m,n,a,lda,d,e,tauq,taup,work,lwork,info);record(9,*m,*n,*lwork==-1,*info);}
