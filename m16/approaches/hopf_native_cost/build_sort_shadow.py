"""Extract unchanged installed-revision numeric timsort; count comparator calls."""
import hashlib,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'native_source/timsort.hpp'
EXPECTED=''
PRELUDE=r'''
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <utility>
#include <type_traits>
using npy_intp = int64_t;
#define NPY_UNLIKELY(x) (x)
#define NPY_ENOMEM 12
#define NPY_UNUSED(x) x
#define PyMem_RawRealloc std::realloc
#define PyMem_RawFree std::free
static uint64_t compare_calls=0, scalar_comparisons=0;
struct DoubleTag {
 using type=double;
 static bool less(double a,double b) {
  ++compare_calls; ++scalar_comparisons;
  if(a<b)return true;
  ++scalar_comparisons;
  if(b!=b) {++scalar_comparisons; return a==a;}
  return false;
 }
 static bool greater(double a,double b) {return less(b,a);}
};
struct IntTag {
 using type=int64_t;
 static bool less(int64_t a,int64_t b){++compare_calls; ++scalar_comparisons;return a<b;}
 static bool greater(int64_t a,int64_t b){return less(b,a);}
};
namespace npy {
 template<typename Tag,bool reverse,typename... Args>
 int cmp(Args... args) {
  if constexpr(reverse)return Tag::greater(args...);
  else return Tag::less(args...);
 }
}
'''
WRAPPERS=r'''
extern "C" int stable_double(double *values,int64_t *indices,int64_t n,uint64_t *counts) {
 compare_calls=scalar_comparisons=0;
 int result=atimsort_<DoubleTag,double,false>(values,indices,n);
 counts[0]=compare_calls;counts[1]=scalar_comparisons;return result;
}
extern "C" int stable_int64(int64_t *values,int64_t *indices,int64_t n,uint64_t *counts) {
 compare_calls=scalar_comparisons=0;
 int result=atimsort_<IntTag,int64_t,false>(values,indices,n);
 counts[0]=compare_calls;counts[1]=scalar_comparisons;return result;
}
'''
def main():
 raw=SOURCE.read_text();needle='/* For string sorts and generic sort, element comparisons are very expensive,'
 assert raw.count(needle)==1
 numeric=raw.split(needle)[0]
 numeric=re.sub(r'^#include .*\n','',numeric,flags=re.M)
 numeric=re.sub(r'^#define NPY_NO_DEPRECATED_API .*\n','',numeric,flags=re.M)
 out=HERE/'sort_shadow.cpp';assert not out.exists();out.write_text(PRELUDE+numeric+WRAPPERS)
 (HERE/'sort_shadow_extraction.json').write_text(json.dumps({'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'generated_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'changes':['remove original includes/API macro; supply standalone allocation/type shims','retain complete numeric timsort source unchanged','substitute semantically matching comparator with side counters','append two exported wrappers; no source decision/path change'],'scope':'Shadow source comparator census. No replacement of NumPy output, no installed-binary equivalence claim, no control/memory cost included.'},indent=2)+'\n')
if __name__=='__main__':main()
