"""Replace only reviewed fixed-layout matrix terms; no numerical model calls."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent


def main():
    outpath = HERE / 'revised_summary.json'
    assert not outpath.exists(), 'Preserve prior contraction'
    start = time.process_time()
    oldpath = HERE.parent / 'summary.json'
    old = json.loads(oldpath.read_text())
    census = json.loads((HERE.parent / 'frozen/full_summary.json').read_text())
    layout = json.loads((HERE / 'layout_applicability.json').read_text())
    assert layout['complete'] and layout['npy_header']['shape'] == [35098, 4, 4]
    assert layout['npy_header']['fortran_order'] is False
    for key in ('indexed_rays_layout', 'du_layout', 'cofactor_N_allocation'):
        assert layout[key]['C'] and layout[key]['strides'] == [32, 8]
    assert all(r['output']['C'] and r['output']['strides'] == [32, 8]
               for r in layout['synthetic_inverses'])
    assert old['source_sha256'] == layout['source_sha256'] == census['pins']['estT_snapshot.py']
    t, w, n = census['thin'], census['wide'], census['cases']
    assert (t, w, n) == (17955, 17143, 35098) and t+w == n
    prices = dict(thin_DU=dict(old_scalar_expression=28000, native_body=72000),
                  selftranspose=dict(old_scalar_expression=112, native_body=170,
                                     selected_guard_envelope=11, replacement=181))
    branches = {}
    for name, copies, selfcount, ducount in [('thin', t, 1, 1), ('wide', w, 2, 0)]:
        b = old['branches'][name]
        removed = ducount*28000 + selfcount*112
        body = ducount*72000 + selfcount*170
        guards = selfcount*11
        revised = b['hybrid_subtotal'] - removed + body + guards
        branches[name] = dict(copies=copies, old_hybrid=b['hybrid_subtotal'],
                              removed_scalar_matrix_terms=removed,
                              replacement_matrix_body=body,
                              replacement_guard_envelope=guards,
                              revised_hybrid=revised, remaining_visible=b['visible']-removed)
    revised = sum(r['copies']*r['revised_hybrid'] for r in branches.values())
    removed = t*28000+(n+w)*112
    body = t*72000+(n+w)*170
    guards = (n+w)*11
    assert revised == old['hybrid_subtotal']-removed+body+guards
    assert revised == 3572147383
    assert body == 1301640970 and body > old['old_legacy_mass']
    ratio = Fraction(body, old['old_legacy_mass'])
    result = dict(complete=True, prices=prices, branches=branches,
                  old_summary_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),
                  old_hybrid=old['hybrid_subtotal'], removed_scalar_matrix_terms=removed,
                  executed_matrix_body_subset=body, selected_matrix_guard_envelopes=guards,
                  revised_hybrid_subtotal=revised,
                  model_free_runtime_constants_unchanged=old['model_free_runtime_constants'],
                  revised_hybrid_plus_those_constants=revised+old['model_free_runtime_constants'],
                  old_legacy_mass=old['old_legacy_mass'],
                  body_over_legacy_exact_fraction=[ratio.numerator, ratio.denominator],
                  source_sha256=old['source_sha256'], full_census_review='4cc3fe39',
                  initial_hybrid_review='26fd4e11', matrix_dispatch_review='155b3b18',
                  child_dispatch_review='0cfdfa39', matrix_body_review='3ed60f55',
                  new_mass_model_matrix_inverse_MC_calls=0,
                  cpu_seconds=time.process_time()-start,
                  complete_native_upper_or_lower=False, same_budget_comparison=False,
                  scope='Fixed saved C-layout replay and pinned one-thread backend only. Executed scalar lanes (FMA=2) differ from the inherited abstract proxy. Matrix-body subset is exact for the reviewed paths; revised hybrid still mixes remaining visible-expression terms with native category envelopes. Reduction/norm/libm/control/integer/memory/cold-setup/geometry and model-free dispatch gaps remain. No runtime, sampler, protocol-wide or estimator-family claim.')
    outpath.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
