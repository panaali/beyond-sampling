"""Reproduce fixed negative/domain failures, preserve frame evidence and node probes."""
import json,time,resource,hashlib,math
from pathlib import Path
import numpy as np
from . import author_snapshot as M
HERE=Path(__file__).parent

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,22));start=time.process_time();rows=[]
    for kind,index in (('quadratic',5826),('nearest',420),('nearest',495),('nearest',10506)):
        rays=np.load(HERE.parent/'class_region_t4'/(kind+'_exact_mass_calls.npz'))['rays'][index];case=dict(kind=kind,index=index,rays=rays.tolist(),rays_sha256=hashlib.sha256(rays.astype('<f8').tobytes()).hexdigest(),determinant=float(np.linalg.det(rays)),probes=[])
        for n in ((None,) if index==420 else (None,20,40,80,160)):
            op=M.Ops();probe=dict(nodes=n)
            try:probe['value']=float(M.mass4_rays(rays,op,n=n))
            except Exception as exc:
                frames=[];tb=exc.__traceback__
                while tb:
                    fr=tb.tb_frame
                    if fr.f_code.co_filename==M.__file__:
                        nums={k:float(v) for k,v in fr.f_locals.items() if k in ('rho0','rs','phis','phi0','den','num','r','det','dphi','scale','score') and isinstance(v,(float,int))}
                        arrays={k:np.asarray(fr.f_locals[k]).tolist() for k in ('m','C','R') if k in fr.f_locals}
                        frames.append(dict(function=fr.f_code.co_name,line=tb.tb_lineno,scalars=nums,arrays=arrays))
                    tb=tb.tb_next
                probe.update(failure_type=type(exc).__name__,failure=str(exc),frames=frames)
            probe['author_ops']=op.n;case['probes'].append(probe)
        rows.append(case)
    out=dict(complete=True,rows=rows,cpu=time.process_time()-start,source_changed=False)
    (HERE/'localization.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
