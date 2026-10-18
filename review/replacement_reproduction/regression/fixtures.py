"""Fixed tiny integer fixtures; no randomness or owner implementation."""
def cases():
    return [
      {'name':'scalar','ids':[0,0,0],'v':[[1],[2],[3]],'y':[[2,-1],[4,-2],[6,-3]]},
      {'name':'two_cells','ids':[2,2,2,9,9,9],'v':[[1,0],[0,1],[1,1],[2,0],[0,2],[2,2]],'y':[[1,3],[2,-1],[3,2],[4,1],[2,5],[6,6]]},
      {'name':'collinear','ids':[0,0,0],'v':[[1,2],[2,4],[-1,-2]],'y':[[1,3],[2,6],[-1,-3]]},
      {'name':'three_cells','ids':[-2,-2,0,0,7,7],'v':[[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]],'y':[[1,2],[2,-1],[3,1],[3,1],[5,0],[4,3]]},
      {'name':'zero_cell','ids':[0,0,1,1],'v':[[0,0],[0,0],[1,0],[0,1]],'y':[[5,-3],[-5,2],[1,2],[2,4]]},
      {'name':'singular_pool','ids':[0,0],'v':[[0,0],[0,0]],'y':[[1],[-1]]}]

def queries(case):
    d=len(case['v'][0]);lo=min(case['ids']);hi=max(case['ids'])
    return [lo,hi,lo-1,hi+1,lo],[case['v'][0],case['v'][-1],[1]*d,[-1]*d,[0]*d]

def guards():return ['cell_dtype','design_dtype','target_rows','nonfinite_design','empty_cells','prediction_dimension','codec_schema','codec_duplicate_ids']
