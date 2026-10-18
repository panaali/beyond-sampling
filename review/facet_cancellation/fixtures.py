"""Fixed constructed controls; no random or trained data."""
def fixtures():
    e = [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]
    m = (1,1,0,0)
    f = (1,1,2,0)  # midpoint between m/2 and e2 on the shared face
    def s(rows):
        return {'rays': rows, 'expected_det': None}
    controls = [
        ('orthant4', 4, [s(e)], 4, 4, 0),
        ('split4', 4, [s([e[0],m,e[2],e[3]]),s([m,e[1],e[2],e[3]])], 6, 7, 1),
        ('duplicate4', 4, [s(e),s(e)], 4, 4, 0),
        ('permuted4', 4, [s(e[::-1])], 4, 4, 0),
        ('scaled4', 4, [s([tuple(3*x for x in r) for r in e])], 4, 4, 0),
        ('negative4', 4, [s([tuple(-x for x in r) for r in e])], 4, 4, 0),
        ('nonconforming4', 4, [s([e[0],m,e[2],e[3]]),s([e[1],m,f,e[3]]),s([e[1],f,e[2],e[3]])], 10, 11, 1),
        ('orthant3', 3, [s([(1,0,0),(0,1,0),(0,0,1)])], 3, 3, 0),
    ]
    # Expected counts are nonzero chain facets (including zero anchor determinants), unique, cancelled.
    guards = [
        ('empty', [], 4),
        ('zero_ray', [s([e[0],e[1],e[2],(0,0,0,0)])], 4),
        ('duplicate_ray', [s([e[0],e[0],e[2],e[3]])], 4),
        ('mixed_orthant', [s([e[0],e[1],e[2],(-1,0,0,1)])], 4),
        ('float_coordinate', [s([(1.,0,0,0),e[1],e[2],e[3]])], 4),
        ('bad_saved_det', [{'rays': e, 'expected_det': ['2','1']}], 4),
    ]
    return controls, guards
