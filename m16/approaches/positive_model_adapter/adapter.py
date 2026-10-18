"""Auditable two-stage prototype; no stored models, mass tables or corpus inputs."""
from pathlib import Path
from fractions import Fraction
from collections import Counter
import ast,hashlib,importlib.util,json,math,sys,time,types
import numpy as np
P=Path(__file__).resolve().parent

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def load():
    pins=json.loads((P/'adapter_pins.json').read_text())
    for f,h in pins.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
    engine_path=Path(next(f for f in pins if f.endswith('estT_snapshot.py')));m43=Path(next(f for f in pins if f.endswith('/m43.py')))
    engine_tree=ast.parse(engine_path.read_text());counter_tree=ast.parse(m43.read_text())
    ops_class=next(n for n in counter_tree.body if isinstance(n,ast.ClassDef) and n.name=='Ops')
    removed=[n for n in engine_tree.body if isinstance(n,ast.ImportFrom) and n.module=='m43']
    assert len(removed)==1 and len(removed[0].names)==1 and removed[0].names[0].name=='Ops'
    engine_tree.body=[n for n in engine_tree.body if n not in removed]
    E=types.ModuleType('adapter_engine');E.__file__=str(engine_path)
    exec(compile(ast.Module(body=[ops_class],type_ignores=[]),str(m43),'exec'),E.__dict__)
    exec(compile(engine_tree,str(engine_path),'exec'),E.__dict__)
    guard=module('adapter_guard',P/'guard.py');original,counts,patched=guard.install(E)
    M=module('adapter_selector',P.parent/'positive_h8_cost/run.py')
    old=module('adapter_positive',M.OLD);kernel=module('adapter_inner',M.KERNEL)
    sys.path.insert(0,str(M.RESCUE))
    try:rescue=module('adapter_exact_rescue',M.RESCUE/'run.py')
    finally:sys.path.pop(0)
    radial=module('adapter_radial',P.parent/'radial_t4_reference/radial.py');radial.F=rescue.Tracked
    # Deliberately do not invoke historical M.load/rescue.load corpus readers.
    # All numerical source pins are validated above instead.
    return E,counts,M,old,kernel,rescue,radial

def collect(model,E,guard_counts):
    start=time.process_time();rays=[];E._HALVES.clear();E.STATS.update({k:0 for k in E.STATS});guard_counts.clear();ops=E.Ops()
    def record(R,o):
        assert R.shape==(4,4);rays.append(R.copy());E.STATS['simplices']+=1;return 0.
    E.orthant=record;ignored=E.estT(model,ops,mode='exact')
    assert ignored==1.
    # List retains the16-element copied arrays directly; no second stacking copy.
    return rays,dict(cpu=time.process_time()-start,N=len(rays),copied_float64_elements=16*len(rays),ray_storage_bytes=128*len(rays),array_header_list_storage_unknown=True,stats=dict(E.STATS),guard_counts=dict(guard_counts),inherited_geometry_proxy=ops.n,cold_halves_internal_ops_not_in_proxy=True,complete_geometry_cost=False,ignored_zero_callback_score=ignored)

def evaluate(rays,M,old,kernel,rescue,radial):
    start=time.process_time();N=len(rays);rules=kernel.Rules();rows=[];known=Counter();exact_tolerance=Fraction(1,10**7*N) if N else None;round_down=False
    if N:
        M.TOL=1e-7/N
        if Fraction(M.TOL)>exact_tolerance:M.TOL=math.nextafter(M.TOL,0.);round_down=True
        assert Fraction(M.TOL)<=exact_tolerance
        rescue.TOLERANCE=exact_tolerance
    for i,R in enumerate(rays):
        row=M.one(R,old,kernel,rescue,radial,rules);row['index']=i;rows.append(row)
        known.update(row['center']['setup_selection_counts']);known.update(row.get('repeated_determinant_counts',{}));known.update(row.get('kernel_counts',{}))
        if row['mode']!='returned':break
        assert row['bound']<=M.TOL
    known.update(rules.ledger.dict());known['universal_constant_arithmetic']+=6
    complete=len(rows)==N and all(r['mode']=='returned' for r in rows)
    out=dict(complete=complete,N=N,rows=rows,known_mass_counts=dict(known),known_mass_total=sum(known.values()),cold_rules=rules.events,allocation=dict(float_tolerance=M.TOL if N else None,exact_tolerance=str(exact_tolerance),nextafter_down=round_down,float_divisions=int(N>0),fraction_and_comparison_work_separate=True),aggregation='math.fsum values/bounds; internal implementation work separate',cpu=time.process_time()-start,complete_cost=False,full_floating_certificate=False)
    if complete:out.update(accuracy=1-math.fsum(r['value'] for r in rows),conditional_bound=math.fsum(r['bound'] for r in rows))
    return out
