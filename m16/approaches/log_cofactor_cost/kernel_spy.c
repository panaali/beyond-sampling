#include <stdint.h>
#include <string.h>
typedef int64_t I;
static I events[256][5],count;
static uintptr_t table;
static void *original[11];
static const I offsets[11]={0x328,0x368,0x390,0x398,0x3a0,0x3c8,0x3e8,0x440,0x448,0x478,0x490};
static void event(I kind,I a,I b,I c,I d){if(count<256){events[count][0]=kind;events[count][1]=a;events[count][2]=b;events[count][3]=c;events[count][4]=d;}count++;}
static I iamax(I n,double*x,I inc){event(0,n,inc,0,0);return ((I(*)(I,double*,I))original[0])(n,x,inc);}
static double dot(I n,double*x,I ix,double*y,I iy){event(1,n,ix,iy,0);return ((double(*)(I,double*,I,double*,I))original[1])(n,x,ix,y,iy);}
#define SCALE(NAME,IDX) static int NAME(I n,I d0,I d1,double alpha,double*x,I ix,double*y,I iy,double*z,I iz){event(IDX,n,ix,iy,iz);return ((int(*)(I,I,I,double,double*,I,double*,I,double*,I))original[IDX])(n,d0,d1,alpha,x,ix,y,iy,z,iz);}
SCALE(scal,2) SCALE(swap,3)
static int gemv(I m,I n,I dummy,double alpha,double*A,I lda,double*x,I ix,double*y,I iy,double*buf){event(4,m,n,ix,iy);return ((int(*)(I,I,I,double,double*,I,double*,I,double*,I,double*))original[4])(m,n,dummy,alpha,A,lda,x,ix,y,iy,buf);}
static int gemm(I m,I n,I k,double alpha,double*A,double*B,double*C,I ldc){event(5,m,n,k,ldc);return ((int(*)(I,I,I,double,double*,double*,double*,I))original[5])(m,n,k,alpha,A,B,C,ldc);}
static int copy(I m,I n,double*A,I lda,double*B){event(6,m,n,lda,0);return ((int(*)(I,I,double*,I,double*))original[6])(m,n,A,lda,B);}
#define TRSM(NAME,IDX) static int NAME(I m,I n,I k,double alpha,double*A,double*B,double*C,I ldc,I offset){event(IDX,m,n,k,offset);return ((int(*)(I,I,I,double,double*,double*,double*,I,I))original[IDX])(m,n,k,alpha,A,B,C,ldc,offset);}
TRSM(upper,7) TRSM(lower,8)
#define PACK(NAME,IDX) static int NAME(I m,I n,double*A,I lda,I offset,double*B){event(IDX,m,n,lda,offset);return ((int(*)(I,I,double*,I,I,double*))original[IDX])(m,n,A,lda,offset,B);}
PACK(upack,9) PACK(lpack,10)
void kernel_install(uintptr_t address){void *wrappers[11]={(void*)iamax,(void*)dot,(void*)scal,(void*)swap,(void*)gemv,(void*)gemm,(void*)copy,(void*)upper,(void*)lower,(void*)upack,(void*)lpack};table=address;for(int i=0;i<11;i++){void **p=(void**)(table+offsets[i]);original[i]=*p;*p=wrappers[i];}}
void kernel_restore(void){for(int i=0;i<11;i++)*(void**)(table+offsets[i])=original[i];}
void kernel_reset(void){count=0;memset(events,0,sizeof(events));}
I kernel_events(I*out){if(count>256)return -count;memcpy(out,events,sizeof(events));return count;}
