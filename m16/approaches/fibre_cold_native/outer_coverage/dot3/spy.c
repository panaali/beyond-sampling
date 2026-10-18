#include <stdint.h>
#include <stddef.h>
_Static_assert(sizeof(long)==8,"ILP64 long required");
typedef double (*dotfn)(long,const double*,long,const double*,long);
struct event {long n,incx,incy;};
static struct event events[1024];
static long count=0;
static int overflow=0,installed=0;
static void **table=NULL;
static dotfn original=NULL;
static double wrapped(long n,const double*x,long incx,const double*y,long incy){
 if(count<1024){events[count].n=n;events[count].incx=incx;events[count].incy=incy;count++;}else overflow=1;
 return original(n,x,incx,y,incy);
}
int install(void **t,void *expected){
 if(installed || !t || t[0x368/8]!=expected)return 0;
 table=t;original=(dotfn)expected;table[0x368/8]=(void*)wrapped;installed=1;return 1;
}
int restore(void){
 if(!installed || !table || table[0x368/8]!=(void*)wrapped)return 0;
 table[0x368/8]=(void*)original;installed=0;table=NULL;original=NULL;return 1;
}
void reset(void){count=0;overflow=0;}
long event_count(void){return count;}
int event_overflow(void){return overflow;}
long event_value(long i,long j){if(i<0 || i>=count || j<0 || j>2)return -999;return ((long*)&events[i])[j];}
