"""T442 pre-registered configurations (fixed before any scoring)."""
SOBOL_SEEDS = [584392059, 460644725, 646627996, 851429358, 947655162, 105847341, 140292349, 308723750, 624882985,
               752329037]  # np.random.default_rng(442).integers(1, 2**30, 10)
FROZEN = ("sobol", 31415)  # T429's frozen T10 rule: rows reused from walk_budget/runs/t10_*.jsonl
SHIFTS = [1, 2, 3, 4, 5, 6]  # lattice shift seeds; lat and latb share them (paired)
HALTON_SEEDS = [1, 2, 3, 4]
CONFIGS = ([("sobol", s) for s in SOBOL_SEEDS] + [("lat", s) for s in SHIFTS] + [("latb", s) for s in SHIFTS]
           + [("lat0", 0)] + [("halton", s) for s in HALTON_SEEDS])
BUDGETS = [256000, 1000000, 4000000]
PRIMARY = [1000000, 4000000]
