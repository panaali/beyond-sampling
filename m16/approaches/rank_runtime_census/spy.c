#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int64_t I;
typedef void (*gesdd_fn)(char*,I*,I*,double*,I*,double*,double*,I*,double*,I*,double*,I*,I*,I*);
static void record(I,I,I,I,I);
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
 record(0,*m,*n,*lwork==-1,*info!=0);
 stats[8]+=(*info==0);stats[9]+=(*info>0);stats[10]+=(*info<0);stats[11]=*info;
}

static I nested[12];
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

}

void scipy_dlasq3_64_(I*i0,I*n0,double*z,I*pp,double*dmin,double*sigma,double*desig,double*qmax,I*nfail,I*iter,I*ndiv,void*ieee,I*ttype,double*dmin1,double*dmin2,double*dn,double*dn1,double*dn2,double*g,double*tau){
 typedef void(*fn)(I*,I*,double*,I*,double*,double*,double*,double*,I*,I*,I*,void*,I*,double*,double*,double*,double*,double*,double*,double*);
 static fn real=NULL;if(!real)real=(fn)resolve("scipy_dlasq3_64_");I before=*iter;I active_before=*n0-*i0+1;I parity_before=*pp;nested[10]++;
 real(i0,n0,z,pp,dmin,sigma,desig,qmax,nfail,iter,ndiv,ieee,ttype,dmin1,dmin2,dn,dn1,dn2,g,tau);
 record(1,active_before,parity_before,0,0);I delta=*iter-before;nested[8]+=delta;if(delta>nested[9])nested[9]=delta;nested[11]+=(delta<0);
}

#define ROWS 1024
static I records[ROWS][6];static I used,overflow;
static void record(I id,I m,I n,I query,I failed){
 for(I i=0;i<used;i++)if(records[i][0]==id&&records[i][1]==m&&records[i][2]==n&&records[i][3]==query){records[i][4]++;records[i][5]+=failed;return;}
 if(used==ROWS){overflow++;return;}I *r=records[used++];r[0]=id;r[1]=m;r[2]=n;r[3]=query;r[4]=1;r[5]=failed;
}
void census_reset(void){memset(records,0,sizeof(records));used=0;overflow=0;audit_reset();nested_reset();}
I census_size(void){return used;}
I census_overflow(void){return overflow;}
void census_get(I*out){memcpy(out,records,used*6*sizeof(I));}

void scipy_dgeqrf_64_(I*m,I*n,double*a,I*lda,double*tau,double*work,I*lwork,I*info){typedef void(*fn)(I*,I*,double*,I*,double*,double*,I*,I*);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dgeqrf_64_");real(m,n,a,lda,tau,work,lwork,info);record(2,*m,*n,*lwork==-1,*info!=0);}
void scipy_dgelqf_64_(I*m,I*n,double*a,I*lda,double*tau,double*work,I*lwork,I*info){typedef void(*fn)(I*,I*,double*,I*,double*,double*,I*,I*);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dgelqf_64_");real(m,n,a,lda,tau,work,lwork,info);record(3,*m,*n,*lwork==-1,*info!=0);}
void scipy_dgebrd_64_(I*m,I*n,double*a,I*lda,double*d,double*e,double*tauq,double*taup,double*work,I*lwork,I*info){typedef void(*fn)(I*,I*,double*,I*,double*,double*,double*,double*,double*,I*,I*);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dgebrd_64_");real(m,n,a,lda,d,e,tauq,taup,work,lwork,info);record(4,*m,*n,*lwork==-1,*info!=0);}
void scipy_dgeqr2_64_(I*m,I*n,double*a,I*lda,double*tau,double*work,I*info){typedef void(*fn)(I*,I*,double*,I*,double*,double*,I*);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dgeqr2_64_");real(m,n,a,lda,tau,work,info);record(5,*m,*n,0,*info!=0);}
void scipy_dgelq2_64_(I*m,I*n,double*a,I*lda,double*tau,double*work,I*info){typedef void(*fn)(I*,I*,double*,I*,double*,double*,I*);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dgelq2_64_");real(m,n,a,lda,tau,work,info);record(6,*m,*n,0,*info!=0);}
void scipy_dgebd2_64_(I*m,I*n,double*a,I*lda,double*d,double*e,double*tauq,double*taup,double*work,I*info){typedef void(*fn)(I*,I*,double*,I*,double*,double*,double*,double*,double*,I*);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dgebd2_64_");real(m,n,a,lda,d,e,tauq,taup,work,info);record(7,*m,*n,0,*info!=0);}
void scipy_dlarfg_64_(I*n,double*alpha,double*x,I*incx,double*tau){typedef void(*fn)(I*,double*,double*,I*,double*);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dlarfg_64_");real(n,alpha,x,incx,tau);record(8,*n,*incx,0,0);}
double scipy_dnrm2_64_(I*n,double*x,I*incx){typedef double(*fn)(I*,double*,I*);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dnrm2_64_");double answer=real(n,x,incx);record(9,*n,*incx,0,0);return answer;}
void scipy_dlasq5_64_(I*i0,I*n0,double*z,I*pp,double*tau,double*sigma,double*dmin,double*dmin1,double*dmin2,double*dn,double*dnm1,double*dnm2,void*ieee,double*eps){typedef void(*fn)(I*,I*,double*,I*,double*,double*,double*,double*,double*,double*,double*,double*,void*,double*);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dlasq5_64_");real(i0,n0,z,pp,tau,sigma,dmin,dmin1,dmin2,dn,dnm1,dnm2,ieee,eps);record(10,*n0-*i0+1,*pp,0,0);}
void scipy_dlasq6_64_(I*i0,I*n0,double*z,I*pp,double*dmin,double*dmin1,double*dmin2,double*dn,double*dnm1,double*dnm2){typedef void(*fn)(I*,I*,double*,I*,double*,double*,double*,double*,double*,double*);static fn real=NULL;if(!real)real=(fn)resolve("scipy_dlasq6_64_");real(i0,n0,z,pp,dmin,dmin1,dmin2,dn,dnm1,dnm2);record(11,*n0-*i0+1,*pp,0,0);}
