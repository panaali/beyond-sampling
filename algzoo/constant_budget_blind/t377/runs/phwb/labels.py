import numpy as np

# Notation used in formulas: M8, s8 = largest and second-largest of x0..x8; p8 = position of M8;
# D = M8 - s8; mx(a,b) = max of xa..xb; sc(a,b) = second-largest of xa..xb. Every formula is ReLU(linear).

LABELS = {
 0: ("A", "Gap between the largest and second-largest of x0..x8, weighted heavily when that largest sits near x3 (x1..x5) and near zero/negative when it sits at x6..x7; plus a small second-largest-value term."),
 1: ("A", "How far x9 falls below the maximum of x0..x8 (zero if x9 is above it); the remembered maximum is boosted when it sat at x2..x4 and pulled toward the runner-up when it sat at x6..x7."),
 2: ("A", "Memory of the running maximum of x0..x8 (ignores x9), with extra weight when that maximum lies in x4..x8."),
 3: ("D", "Unclear position-set code: grows with max(x0,x1), max(x1..x8) and x6, shrinks with max(x5..x7); roughly, larger when the top values sit at x0, x1 or x6 and smaller at x5 or x7."),
 4: ("A", "Fires when x9 is a new record: proportional to how much x9 exceeds about 0.9 times the maximum of x0..x8; zero otherwise."),
 5: ("A", "Fires when x8 is large relative to earlier inputs: x8 minus the second-largest values among x0..x7 (sec of x2..x7 and x0..x5), i.e. x8 is among the top two so far."),
 6: ("A", "How far x9 falls below the maximum of x0..x7 (extra weight on x3..x6); zero when x9 is at least as large as that earlier maximum."),
 7: ("A", "How far x9 falls below the overall maximum of x0..x9 (extra weight on x5..x9); zero when x9 itself is the maximum."),
 8: ("A", "How far x9 falls below the maximum of x0..x8, where that remembered maximum counts more when it sat early (x0..x3) and drifts toward the runner-up when it sat late."),
 9: ("A", "Weak detector that x8 is the largest of x2..x8 (a new recent record), scaled by x8's value and reduced by the second-largest of x0..x5."),
 10: ("A", "How much the maximum of x0..x8 exceeds everything in x0..x5 (mostly x1..x4): large when the maximum sits at x6..x8 (or x0/x5), zero when it sits at x2..x3."),
 11: ("A", "How much the maximum of x0..x8 exceeds everything in x1..x6 (mostly x2..x5): large when the maximum sits at x0, x7 or x8, zero when it sits at x3..x5."),
 12: ("A", "Position-of-maximum code: large when the maximum of x0..x8 sits early (x0..x2) or at x7..x8, suppressed when it sits at x3..x6 or when x0..x6 holds a strong runner-up."),
 13: ("C", "Mostly the maximum of x0..x8, plus bonuses when max(x4,x5) or max(x0,x1) is large, minus the second-largest of x2..x7; a loose position-set code that is only partly understood."),
 14: ("A", "Gap between the largest and second-largest of x0..x8 weighted by a bump peaking when the largest sits at x3 (negative near x7), plus the maximum itself, minus the runner-up of x0..x7 and max(x0,x1)."),
 15: ("A", "Early-maximum detector: grows with max(x0..x4) and max(x0..x3), reduced by max(x3,x4) and max(x6,x7); large when the largest values sit in x0..x2."),
}

CONSTANTS = {0: 5, 1: 4, 2: 3, 3: 5, 4: 3, 5: 4, 6: 4, 7: 4, 8: 5, 9: 4, 10: 5, 11: 5, 12: 5, 13: 5, 14: 5, 15: 5}

STRUCT = {
 0: "features searched: D, D*cos(w(p8-phi)) with w=0.8, phi=-1 from grid; s8, sc(1,7) from window library",
 1: "cos tuning w=0.95, phi=6.5 searched on grid",
 2: "window x4..x8 chosen from all contiguous windows",
 3: "windows (1,8),(0,1),(6,6),(5,7) chosen greedily from window-max/second library",
 4: "",
 5: "windows for second-largest (2,7),(0,5) chosen greedily",
 6: "windows (0,7),(3,6) chosen from window library",
 7: "window (5,9) chosen from window library",
 8: "cos tuning w=0.6, phi=1.25 searched on grid",
 9: "windows (2,8),(0,5) chosen greedily",
 10: "windows (1,4),(0,5),(2,3) chosen greedily",
 11: "windows (2,5),(3,6),(1,4) chosen greedily",
 12: "windows (0,3),(3,6), second-largest window (0,6) chosen greedily",
 13: "windows (4,5),(0,1), second-largest window (2,7) chosen greedily",
 14: "cos tuning w=0.8, phi=3 from grid; windows sc(0,7), mx(0,1) chosen greedily",
 15: "windows (0,4),(3,4),(6,7),(0,3) chosen greedily",
}

_C = {0: [13.862, -13.387, 8.988, -6.57, 0.327], 1: [9.478, -7.467, -10.5, 0.364], 2: [13.171, 4.322, 0.401],
      3: [3.06, 3.581, 1.832, -1.944, 1.472], 4: [9.975, -9.039, -0.076], 5: [12.959, -5.661, -3.827, -0.565],
      6: [7.147, 3.611, -10.96, 0.134], 7: [10.179, 2.565, -13.32, 0.18], 8: [12.188, 9.025, -4.354, -12.268, 0.103],
      9: [15.086, -13.345, -0.993, -0.439], 10: [21.458, -9.126, -6.949, -3.572, -0.632],
      11: [19.962, -8.33, -5.958, -3.702, 0.013], 12: [13.133, 13.253, -8.9, -9.819, 0.319],
      13: [5.333, 3.172, 1.544, -2.896, 0.879], 14: [14.352, 12.179, -5.816, -2.512, -0.083],
      15: [8.157, -5.145, -2.545, 6.143, -0.284]}


def _mx(x, a, b):
    return x[:, a:b + 1].max(1)


def _sc(x, a, b):
    return np.sort(x[:, a:b + 1], 1)[:, -2]


def _features(i, x):
    one = np.ones(len(x))
    P = x[:, :9]
    s = np.sort(P, 1)
    M8, s8 = s[:, -1], s[:, -2]
    D = M8 - s8
    p8 = P.argmax(1)
    x9, x8 = x[:, 9], x[:, 8]
    if i == 0: return [D, D * np.cos(0.8 * (p8 + 1)), s8, _sc(x, 1, 7), one]
    if i == 1: return [M8, D * np.cos(0.95 * (p8 - 6.5)), x9, one]
    if i == 2: return [M8, _mx(x, 4, 8), one]
    if i == 3: return [_mx(x, 1, 8), _mx(x, 0, 1), x[:, 6], _mx(x, 5, 7), one]
    if i == 4: return [x9, M8, one]
    if i == 5: return [x8, _sc(x, 2, 7), _sc(x, 0, 5), one]
    if i == 6: return [_mx(x, 0, 7), _mx(x, 3, 6), x9, one]
    if i == 7: return [_mx(x, 0, 9), _mx(x, 5, 9), x9, one]
    if i == 8: return [M8, D * np.cos(0.6 * (p8 - 1.25)), D, x9, one]
    if i == 9: return [x8, _mx(x, 2, 8), _sc(x, 0, 5), one]
    if i == 10: return [M8, _mx(x, 1, 4), _mx(x, 0, 5), _mx(x, 2, 3), one]
    if i == 11: return [M8, _mx(x, 2, 5), _mx(x, 3, 6), _mx(x, 1, 4), one]
    if i == 12: return [M8, _mx(x, 0, 3), _sc(x, 0, 6), _mx(x, 3, 6), one]
    if i == 13: return [M8, _mx(x, 4, 5), _mx(x, 0, 1), _sc(x, 2, 7), one]
    if i == 14: return [M8, D * np.cos(0.8 * (p8 - 3)), _sc(x, 0, 7), _mx(x, 0, 1), one]
    if i == 15: return [_mx(x, 0, 4), _mx(x, 3, 4), _mx(x, 6, 7), _mx(x, 0, 3), one]


def f(i, x):
    x = np.asarray(x, dtype=float)
    A = np.stack(_features(i, x), 1)
    return np.maximum(A @ np.array(_C[i]), 0.0)
