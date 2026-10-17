"""Owned native primitive and private StagePieces substitution only."""
import ctypes,hashlib,json,resource,subprocess,time,types
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
HERE=Path(__file__).resolve().parent


def build():
    folder=HERE/'build';folder.mkdir(exist_ok=True)
    target=folder/'gaussian_basis.so'
    before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.perf_counter()
    command=['c++','-O2','-std=c++17','-shared','-fPIC','-ffp-contract=off',str(HERE/'kernel.cpp'),'-o',str(target)]
    run=subprocess.run(command,text=True,capture_output=True,check=True)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    data=dict(command=command,wall_seconds=time.perf_counter()-start,child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,source_sha256=hashlib.sha256((HERE/'kernel.cpp').read_bytes()).hexdigest(),stdout=run.stdout,stderr=run.stderr)
    (HERE/'build.json').write_text(json.dumps(data,indent=2)+'\n')
    return data


class NativeBasis:
    def __init__(self,stats):
        start=time.process_time()
        self.library=ctypes.CDLL(str(HERE/'build/gaussian_basis.so'))
        self.function=self.library.gaussian_basis
        array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        count=np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS')
        self.function.argtypes=[ctypes.c_int64]+[array]*10+[count]
        self.function.restype=ctypes.c_int
        self.nodes,self.weights=leggauss(12)
        stats['native_library_loads']+=1;stats['native_gauss12_rules_constructed']+=1
        stats['native_library_rule_setup_cpu_seconds']+=time.process_time()-start

    def integrate(self,lo,hi,s,t,ds,dt,stats):
        arrays=[np.ascontiguousarray(x,dtype=float) for x in (lo,hi,s,t,ds,dt)]
        n=len(arrays[0])
        if any(x.shape!=(n,) for x in arrays):raise ValueError('native basis arrays must be flat and equal-length')
        out=np.empty((n,4));bounds=np.empty(n);counts=np.zeros(5,dtype=np.uint64)
        status=self.function(n,*arrays,self.nodes,self.weights,out,bounds,counts)
        if status:raise ArithmeticError('native Gaussian basis invalid/nonfinite input or outside supported [-6,6] domain')
        total,fallback,badmom,badvalue,nodes=map(int,counts)
        stats['polynomial_intervals']+=total
        stats['polynomial_centered_moment_triples']+=total
        stats['polynomial_cdf_values']+=2*total
        stats['polynomial_endpoint_density_exponentials']+=2*total
        stats['polynomial_expm1_calls']+=total
        stats['polynomial_basis_mixes']+=3*total+nodes
        stats['polynomial_positive_fallback_intervals']+=fallback
        stats['polynomial_positive_fallback_nodes']+=nodes
        stats['polynomial_moment_screen_failures']+=badmom
        stats['polynomial_value_screen_failures']+=badvalue
        stats['native_kernel_calls']+=1
        stats['native_erfc_calls']+=2*total
        stats['native_exp_calls']+=2*total+nodes+2*fallback
        stats['native_expm1_calls']+=total
        stats['native_output_scalar_values']+=5*n
        return out,bounds

    def stage_class(self,original):
        # Only StagePieces creates eye(4) basis corners; no general continuation API.
        def interval(lo,hi,corners,s,t,ds,dt,stats):
            if any(v.shape!=(len(lo),4) for v in corners):raise ValueError('native StagePieces requires four basis corners')
            return self.integrate(lo,hi,s,t,ds,dt,stats)
        namespace=dict(original.__init__.__globals__);namespace['interval_integrals']=interval
        init=types.FunctionType(original.__init__.__code__,namespace,original.__init__.__name__,original.__init__.__defaults__,original.__init__.__closure__)
        matrices=types.FunctionType(original.matrices.__code__,namespace,original.matrices.__name__,original.matrices.__defaults__,original.matrices.__closure__)
        return type('NativeStagePieces',(original,),{'__init__':init,'matrices':matrices})
