"""Per-configuration table on the base model of M_16,T: python config_table.py T SIMPLE,AUGMENT uncond,cell (run from repo root)."""
import sys, time; sys.path.insert(0,"m16/approaches/arc_cumulants"); sys.path.insert(0,"m16")
import arc_kprop as A, leaderboard as LB, json
H,T=16,int(sys.argv[1])
name,m=LB._base(H,T); tr=json.load(open(LB.truths_path(H,T)))[name]["acc"]
fpe=LB.fpe_flops(H,T)
for kind in sys.argv[2].split(","):
  for K in [1,2,3,4]:
    for mode in sys.argv[3].split(","):
      t0=time.time()
      try:
        e,raw,adj=(A.estimate_uncond if mode=="uncond" else A.estimate_cell)(m,K,kind)
        print(f"{kind:8s} K={K} {mode:6s} est={e:.5f} truth={tr:.5f} bias={100*(e-tr):+.3f}pp cost={raw/fpe:.0f}FPE (sym-adj {adj/fpe:.0f}) {time.time()-t0:.1f}s",flush=True)
      except Exception as ex:
        print(kind,K,mode,"ERROR",repr(ex)[:300],flush=True)
