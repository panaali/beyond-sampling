import numpy as np

def basics(x):
    x = np.asarray(x, dtype=float); N = x.shape[0]
    M = np.maximum.accumulate(x, 1)
    G = np.zeros((N, 10)); g = np.zeros((N, 10)); u = np.zeros((N, 10)); T = np.zeros((N, 10))
    T[:, 0] = x[:, 0]
    for t in range(1, 10):
        G[:, t] = np.maximum(M[:, t-1] - x[:, t], 0)
        s = -np.sort(-x[:, :t+1], 1); T[:, t] = s[:, :2].mean(1)
        g[:, t] = np.maximum(T[:, t-1] - x[:, t], 0)
        u[:, t] = np.maximum(x[:, t] - T[:, t-1], 0)
    srt = -np.sort(-x, 1); s9 = -np.sort(-x[:, :9], 1)
    return dict(x=x, M=M, G=G, g=g, u=u, T=T, top2sum=srt[:, :2].sum(1), top=srt[:, 0], sec=srt[:, 1], top4_08=s9[:, :4].sum(1))
def design(i, b):
    x, G, g, u, T = b['x'], b['G'], b['g'], b['u'], b['T']
    if i == 0: return [G[:, 3:6].sum(1), T[:, 8], g[:, 8], g[:, 1:6].sum(1), b['top4_08']]
    if i == 1: return [g[:, 4:7].sum(1), x[:, 4:6].sum(1), g[:, 5], G[:, 2:7].sum(1), u[:, 7]]
    if i == 2: return [G[:, 3:10].sum(1), x[:, 7:9].sum(1), g[:, 1:8].sum(1), G[:, 4], g[:, 3:6].sum(1)]
    if i == 3: return [G[:, 3:6].sum(1), g[:, 7], x[:, 4], g[:, 5], x[:, 1:5].sum(1)]
    if i == 4: return [g[:, 8], g[:, 9]]
    if i == 5: return [T[:, 8], x[:, 9]]
    if i == 6: return [b['top'] - x[:, 4], b['M'][:, 4], G[:, 2], b['sec'] - x[:, 4], x[:, 8]]
    if i == 7: return [G[:, 4:7].sum(1), x[:, 9], g[:, 5], T[:, 8], T[:, 2]]

COEF = {0: [0.9487, 5.9806, -1.3724, 0.8043, -1.0169], 1: [1.5192, 0.6986, -0.9031, 0.17, 0.8536], 2: [0.3529, 0.656, 0.2385, -0.5507, 0.4027], 3: [1.0502, -2.7644, 1.0142, 0.9145, -0.2812], 4: [1.3319, 0.2255], 5: [1.0769, -1.1952], 6: [1.1143, 1.6447, 0.7137, 0.9037, 0.3145], 7: [1.0675, 1.3683, 1.7921, 1.5574, 1.2112]}

LABELS = {0: ('B', 'Overall size of the top values (mean of the two largest of x0..x8, T8) plus accumulated shortfalls of x3,x4,x5 below the running maximum of earlier inputs; reduced when x8 falls short of the earlier top-two mean.'), 1: ('B', 'Accumulated shortfall of x4,x5,x6 below the mean of the two largest earlier inputs (strongest for x6), plus the sum x4+x5; raised when x7 exceeds the earlier top-two mean.'), 2: ('B', 'Broad magnitude signal: total shortfall of x3..x9 below the running maximum of earlier inputs, plus x7+x8, plus shortfalls of x1..x7 below the earlier top-two mean (extra weight on x3..x5).'), 3: ('B', 'Fires when x7 is NOT below the mean of the two largest of x0..x6 (silenced by that shortfall), given accumulated shortfalls of x3,x4,x5 below the running maximum; increases with x4.'), 4: ('A', 'Shortfall of x8 below the mean of the two largest of x0..x7 (zero when x8 is a new top value), plus a small contribution from the same shortfall of x9 relative to x0..x8.'), 5: ('A', 'How far x9 falls below the mean of the two largest of x0..x8: ReLU(a*T8 - b*x9), with a~1.08, b~1.2; zero when x9 is (near) a new maximum.'), 6: ('B', 'How far x4 lies below the overall maximum and below the second-largest value of x0..x9, plus the running maximum of x0..x4; small when x4 is the largest or second-largest input.'), 7: ('B', 'Accumulated shortfall of x4,x5,x6 below the running maximum of earlier inputs (extra for x5), plus x9, plus the size of the top values (top-two means of x0..x8 and of x0..x2).')}

CONSTANTS = {0: 5, 1: 5, 2: 5, 3: 5, 4: 2, 5: 2, 6: 5, 7: 5}

STRUCT = {0: 'features: sum_{t=3..5} relu(max(x0..x_{t-1})-x_t), T8, g8, sum_{t=1..5} g_t, sum of top 4 of x0..x8; chosen by greedy search over window sums', 1: 'features: g4+g5+g6, x4+x5, g5, sum_{t=2..6} relu(running-max gap), relu(x7-T6); greedy search', 2: 'features: sum_{t=3..9} running-max gap, x7+x8, sum_{t=1..7} g_t, gap at x4, g3+g4+g5; greedy search', 3: 'features: sum_{t=3..5} running-max gap, g7, x4, g5, x1+..+x4; greedy search', 4: 'features g8, g9 (g_t = relu(T_{t-1}-x_t), T_k = mean of 2 largest of x0..xk); greedy search', 5: 'features T8 and x9; chose top-two mean as the level after comparing max/top2/top3/top4', 6: 'features: max(x)-x4, max(x0..x4), running-max gap at x2, secondmax(x)-x4, x8; greedy search', 7: 'features: sum_{t=4..6} running-max gap, x9, g5, T8, T2; greedy search'}

def f(i, x):
    """Predict h_10[i]: ReLU of a linear combination (no intercept) of the declared features."""
    b = basics(x)
    B = np.stack(design(i, b), 1)
    return np.maximum(B @ np.array(COEF[i]), 0.0)
