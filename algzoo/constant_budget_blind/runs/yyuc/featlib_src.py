def _feat(name, x):
    """Interpretable features of x (N,10). P1>=P2>=P3 are the sorted values of the prefix x0..x8;
    pa, pb, pc are their positions. 'maxS' = max over positions in S; '2ndS' = second largest over S;
    'P1@S' = P1 if its position pa is in S else 0 (same for P2@S with pb, P3@S with pc)."""
    pre = x[:, :9]
    o = np.argsort(-pre, axis=1)
    srt = np.take_along_axis(pre, o, axis=1)
    if name in ('P1', 'P2', 'P3', 'P4'):
        return srt[:, int(name[1]) - 1]
    if name == 'M7':
        return x[:, :8].max(axis=1)
    if name[0] == 'x':
        return x[:, int(name[1:])]
    if name.startswith('max'):
        S = [int(c) for c in name[3:]]
        return x[:, S].max(axis=1)
    if name.startswith('2nd'):
        S = [int(c) for c in name[3:]]
        return -np.sort(-x[:, S], axis=1)[:, 1]
    if name[0] == 'P' and '@' in name:
        r = int(name[1]) - 1
        S = [int(c) for c in name.split('@')[1]]
        return srt[:, r] * np.isin(o[:, r], S)
    raise ValueError(name)
