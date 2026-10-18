"""Offline SM80 code-generation discriminator only; never launches a CUDA kernel."""
import os
assert os.environ.get('TRITON_INTERPRET')!='1'
import hashlib,json,resource,time,traceback
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
    out=P/'compile_results.json';assert not out.exists(),'refuse overwrite'
    start=time.process_time();result={'complete':False,'pass_all':False,'rows':[],'gpu_launches':0}
    def save():
        result['cpu_seconds']=time.process_time()-start
        tmp=P/'compile.tmp';tmp.write_text(json.dumps(result,indent=2,default=str)+'\n');tmp.replace(out)
    try:
        gate=json.loads((P/'compile_gate.json').read_text());assert gate['ready']
        mandatory=[str(P/f) for f in ('compile_check.py','compile_scope.json','fused_step.py','validation.json')]
        assert all(f in gate['pins'] for f in mandatory)
        for f,h in gate['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
        review=json.loads(Path(gate['review']).read_text());assert review['complete'] and review['pass_all'] and review['execution_allowed']
        assert gate['review'] in gate['pins']
        for f,h in review['pins'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
        validation=json.loads((P/'validation.json').read_text());assert validation['complete'] and validation['pass_all']
        assert resource.getrlimit(resource.RLIMIT_CPU)==(105,110) and resource.getrlimit(resource.RLIMIT_AS)==(2**31,2**31)
        import triton
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        from fused_step import _step
        for batch,warps in ((32,8),(1024,16)):
            row={'B':batch,'warps':warps,'arch':80,'status':'started'};result['rows'].append(row);save()
            constexpr={'B':batch,'BB':batch,'T':10,'FLOOR':.01,'WD':.001,'B1':.9,'B2':.999,'EPS':1e-8,'DIAGNOSTICS':False}
            signature={n:('*i64' if n=='Y' else '*fp64' if n in ('STEPS','NSTEPS','WARM','LR0') else '*fp32') for n in _step.arg_names if n not in constexpr}
            kernel=triton.compile(ASTSource(_step,signature,constexpr),target=GPUTarget('cuda',80,32),options={'num_warps':warps,'enable_fp_fusion':False})
            ptx=kernel.asm['ptx'];(P/f'compile_B{batch}.ptx').write_text(ptx)
            row.update(status='compiled',metadata=dict(kernel.metadata._asdict()),ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),cubin_sha256=hashlib.sha256(kernel.asm['cubin']).hexdigest(),ptx_bytes=len(ptx));save()
        result['complete']=result['pass_all']=True;save()
    except BaseException:
        result['complete']=result['pass_all']=False;result['failure']=traceback.format_exc();save();raise
if __name__=='__main__':main()
