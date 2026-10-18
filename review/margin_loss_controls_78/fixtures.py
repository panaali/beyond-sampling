"""Literal fixture inventory; functions are invoked only by the gated runner."""
def cases():
    base = dict(x=[[1, 0], [0, 1]], hidden=[[2, 1], [1, 3]], penalty=[1, 2],
                readout=[[1, 0, -1], [0, 1, 1]],
                query=[[1, 0], [0, 1], [1, 1], [-1, 1], [0, 0]],
                expected_hidden=[[1, '1/2'], ['1/3', 1]],
                expected_logits=[[1, '1/2', '-1/2'], ['1/3', 1, '2/3']],
                expected_prediction=[[1, '1/2', '-1/2'], ['1/3', 1, '2/3'],
                                     ['4/3', '3/2', '1/6'], ['-2/3', '1/2', '7/6'], [0, 0, 0]],
                expected_labels=[0, 1, 1, 2, 0])
    out = [dict(name='readout_rectangular', kind='commute', **base)]
    rank = dict(base)
    rank.update(readout=[[1, 2], [2, 4]], expected_logits=[[2, 4], ['7/3', '14/3']],
                expected_prediction=[[2, 4], ['7/3', '14/3'], ['13/3', '26/3'], ['1/3', '2/3'], [0, 0]],
                expected_labels=[1, 1, 1, 1, 0])
    out.append(dict(name='readout_rank_deficient', kind='commute', **rank))
    out.append(dict(name='common_mode', kind='common', **base, shift=[4, -3], expected_shift=[2, -1]))
    out.append(dict(name='shared_intercept', kind='intercept', x=[[1, -1], [1, 1]],
                    y=[[0, 2], [2, 0]], penalty=[0, 2], query=[[1, 0], [1, 1], [1, -1]],
                    expected_coefficients=[[1, 1], ['1/2', '-1/2']],
                    expected_prediction=[[1, 1], ['3/2', '1/2'], ['1/2', '3/2']], expected_labels=[0, 0, 1]))
    out.append(dict(name='fixed_cell_pooled_routing', kind='routing', **base,
                    groups=[[0], [1], [0, 1]], routes=[0, 1, 2], route_queries=[[1, 1]]*3,
                    expected_routed=[[1, '1/2', '-1/2'], ['1/3', 1, '2/3'], ['4/3', '3/2', '1/6']]))
    for convention in ('unordered_sum', 'ordered_sum', 'unordered_mean', 'ordered_mean'):
        for mean in (False, True):
            out.append(dict(name=convention+('_sample_mean' if mean else '_sample_sum'), kind='pair',
                            convention=convention, sample_mean=mean, **base))
    out.append(dict(name='unmatched_penalty', kind='unmatched', x=[[1, 0], [0, 1]],
                    y=[[2, 0], [0, 1]], penalty=[1, 3], query=[[1, '7/2']],
                    expected_matched=[[ '1/2', '-1/2'], ['-1/8', '1/8']],
                    expected_unmatched=[['2/3', '-2/3'], ['-1/5', '1/5']],
                    expected_predictions=[['1/16', '-1/16'], ['-1/30', '1/30']], expected_labels=[0, 1]))
    out.append(dict(name='output_specific_maps', kind='specific', x=[[1]], y=[[1, 1]],
                    penalties=[0, 1], sum_penalty=0, query=[[1]], expected=[1, '1/2', 2]))
    out.append(dict(name='singular_solution_convention', kind='singular', x=[[1, 0]], y=[[2, 3]],
                    penalty=[0, 0], query=[[0, 1]], fixed_coefficients=[[2, 3], [0, 0]],
                    alternative_coefficients=[[2, 3], [1, -1]], sum_coefficients=[[5], [7]],
                    expected_queries=[[0, 0], [1, -1], [7]], expected_error='singular system'))
    out.append(dict(name='relu_noncommutation', kind='relu', x=[[1]], hidden=[[2, 1]],
                    penalty=[0], readout=[[1, 0], [0, 1]], query=[[-1]],
                    expected_linear=[[-2, -1]], expected_clipped=[[0, 0]], expected_labels=[1, 0]))
    return out
