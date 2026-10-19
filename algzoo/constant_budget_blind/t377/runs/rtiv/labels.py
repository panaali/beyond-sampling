import numpy as np

# ---------------------------------------------------------------- features
def _feat(x):
    x = np.asarray(x, dtype=float)
    s = -np.sort(-x, 1)
    am = np.argmax(x, 1)
    pm = np.maximum.accumulate(x, 1)
    return dict(x=x, M=s[:, 0], S=s[:, 1], even=(am % 2 == 0).astype(float),
                pm6=pm[:, 6], pm7=pm[:, 7], pm8=pm[:, 8], am=am)

def _relu(z):
    return np.maximum(z, 0.0)

# form name, discrete structure, fitted constants
SPEC = {
#SPEC_START
    0: ('G4', {'P': [0, 4, 5, 7]}, [21.6972, 8.0570, 6.1761, -7.4488, -0.7975]),  # test R2 0.860
    1: ('G3', {'P': [9]}, [16.8998, 14.2436, 7.5079, -0.1951, -2.1268]),  # test R2 0.958
    2: ('G1', {'P': [0, 7, 8]}, [26.0969, 12.5143, 3.6270, -9.6809, -2.0388]),  # test R2 0.900
    3: ('gap', {'t': 9}, [14.1559, -3.5600, 0.5826, 8.6952, 9.3446]),  # test R2 0.919
    4: ('G3', {'P': [1, 5, 7, 8]}, [17.2673, 27.3623, 9.1586, -9.6427, -3.3326]),  # test R2 0.885
    5: ('up', {}, [8.7033, 7.6832]),  # test R2 0.947
    6: ('gap', {'t': 9}, [12.8592, 0.4017, 5.5577, 6.8944, 13.1866]),  # test R2 0.986
    7: ('G3', {'P': [0, 2, 9]}, [29.4466, 11.5434, 12.1429, 0.3417, -8.1058]),  # test R2 0.844
    8: ('gap', {'t': 7}, [14.8268, -2.0678, 2.3858, 8.7250, 11.9718]),  # test R2 0.926
    9: ('Hs', {'Q': [0, 2, 7, 8], 'P': [6]}, [35.0408, 12.4831, -7.1147, 5.7189, -6.0614]),  # test R2 0.848
    10: ('G4', {'P': [3, 7, 8]}, [24.4421, 9.5069, 5.3391, -9.7542, -2.2740]),  # test R2 0.853
    11: ('G1', {'P': [0, 5, 7, 8]}, [8.9783, 19.0701, -7.0758, -0.3191, -4.8617]),  # test R2 0.886
    12: ('gap', {'t': 9}, [7.0547, 4.6034, 10.6674, 1.3712, 12.1664]),  # test R2 0.931
    13: ('Hs', {'Q': [1, 3, 4, 5, 6, 9], 'P': [1, 3, 9]}, [32.5139, 10.9353, 8.0793, 0.1244, -7.9440]),  # test R2 0.863
    14: ('gap14', {}, [6.5530, 3.8961, 6.6091, 2.5805]),  # test R2 0.770
    15: ('H78', {'Q': [5, 9], 'P': [4, 5, 8]}, [16.9644, 20.7404, 7.3522, -10.6718, -1.6200]),  # test R2 0.880
#SPEC_END
}

def _eval(form, struct, c, F):
    x = F['x']
    if form == 'gap':      # ReLU(g*(a1 M + b1 S) + (1-g)*(a2 M + b2 S) - c x_t), g=[argmax even]
        t = struct['t']; g = F['even']
        return _relu(g*(c[0]*F['M'] + c[1]*F['S']) + (1-g)*(c[2]*F['M'] + c[3]*F['S']) - c[4]*x[:, t])
    if form == 'gap14':    # ReLU(a*max(x0..x6) + b*[argmax odd]*M - c*x4 - d*x9)
        return _relu(c[0]*F['pm6'] + c[1]*(1-F['even'])*F['M'] - c[2]*x[:, 4] - c[3]*x[:, 9])
    if form == 'up':
        return _relu(c[0]*x[:, 9] - c[1]*F['pm8'])
    P = struct['P']; rest = [u for u in range(10) if u not in P]
    sp = x[:, P].sum(1); sr = x[:, rest].sum(1)
    if form == 'G1':
        b = [F['pm7'], F['pm8'], F['even']*F['M']]
    elif form == 'G3':
        b = [F['pm7'], F['pm8'], F['S']]
    elif form == 'G4':
        b = [F['pm8'], F['S'], F['even']*F['M']]
    elif form == 'Hs':     # Q = argmax positions that get an extra max weight
        b = [F['pm8'], F['S'], F['M']*np.isin(F['am'], struct['Q'])]
    elif form == 'H78':
        b = [F['pm7'], F['pm8'], F['M']*np.isin(F['am'], struct['Q'])]
    return c[0]*b[0] + c[1]*b[1] + c[2]*b[2] + c[3]*sp + c[4]*sr

def f(i, x):
    form, struct, c = SPEC[i]
    return _eval(form, struct, c, _feat(x))

# Notation used in labels: M = largest of x0..x9, S = second-largest; "max position" = index of M.
LABELS = {
    0: ("A", "Mostly the running maximum of x0..x8, plus some second-largest value S (extra when the max sits at an even position), lowered by the values x0, x4, x5 and x7."),
    1: ("A", "A delayed running maximum: grows with max(x0..x7) and max(x0..x8) (x9 barely counts) plus some of the second-largest value, minus a small penalty on the sum x0+...+x8."),
    2: ("A", "Delayed running maximum (mainly max of x0..x7, partly x8), minus a strong penalty proportional to x0+x7+x8 and a weak one for the other values."),
    3: ("A", "How far x9 falls below a stored record, clipped at zero: the record is the largest value when the max sits at an even position, but roughly the second-largest value when at an odd position."),
    4: ("A", "Delayed running maximum (max of x0..x7 and of x0..x8) plus some second-largest value, minus a penalty proportional to x1+x5+x7+x8 and a weaker one for the other values."),
    5: ("A", "Fires only when x9 is a new record: about 8.7*x9 - 7.7*max(x0..x8), clipped at zero; otherwise silent."),
    6: ("A", "How far x9 lies below the running maximum, about 13*(max - x9), clipped at zero; when the max sits at an odd position the max is blended with the second-largest value."),
    7: ("A", "Delayed running maximum (max of x0..x7, partly x8) plus the second-largest value, minus a penalty on the sum of all values except x0, x2 and x9."),
    8: ("A", "How far x7 lies below a stored record, clipped at zero (zero when x7 is the largest): record is the largest value when the max sits at an even position, mostly the second-largest when odd."),
    9: ("A", "Running maximum of x0..x8 plus some second-largest value, reduced when the max sits at x0, x2, x7 or x8; rises with x6 and falls with the sum of all other values."),
    10: ("A", "Running maximum of x0..x8 plus some second-largest value (extra when the max sits at an even position), minus a strong penalty proportional to x3+x7+x8 and a weak one for others."),
    11: ("A", "Delayed running maximum (max of x0..x8, less when the max sits at an even position), minus a penalty on the sum of x1, x2, x3, x4, x6 and x9."),
    12: ("A", "How far x9 lies below the running maximum, about 12*(max - x9), clipped at zero; when the max sits at an even position the max is blended with the second-largest value."),
    13: ("A", "Running maximum of x0..x8 plus some second-largest value, boosted when the max sits at x1, x3, x4, x5, x6 or x9, minus a penalty on all values except x1, x3, x9."),
    14: ("B", "Mainly how far x4 lies below max(x0..x6), boosted when the overall max sits at an odd position, and lowered by x9; clipped at zero."),
    15: ("A", "Delayed running maximum (max of x0..x7 and x0..x8), boosted when the max sits at x5 or x9, minus a strong penalty proportional to x4+x5+x8 and a weak one for others."),
}
CONSTANTS = {i: len(SPEC[i][2]) for i in range(16)}
STRUCT = {
    0: "form pm8,S,[argmax even]*M + penalty set P={0,4,5,7} (shared weight) + rest; P chosen greedily",
    1: "form pm7,pm8,S + P={9} + rest; P greedy",
    2: "form pm7,pm8,[argmax even]*M + P={0,7,8} + rest; P greedy",
    3: "ReLU gap form, target position t=9, parity split on argmax (even/odd)",
    4: "form pm7,pm8,S + P={1,5,7,8} + rest; P greedy",
    5: "ReLU(a*x9 - b*max(x0..x8))",
    6: "ReLU gap form, target t=9, parity split",
    7: "form pm7,pm8,S + P={0,2,9} + rest; P greedy",
    8: "ReLU gap form, target t=7, parity split",
    9: "form pm8,S,M*[argmax in Q={0,2,7,8}] + P={6} + rest; Q,P greedy",
    10: "form pm8,S,[argmax even]*M + P={3,7,8} + rest; P greedy",
    11: "form pm7,pm8,[argmax even]*M + P={0,5,7,8} + rest; P greedy",
    12: "ReLU gap form, target t=9, parity split",
    13: "form pm8,S,M*[argmax in Q={1,3,4,5,6,9}] + P={1,3,9} + rest; Q,P greedy",
    14: "ReLU(a*max(x0..x6) + b*[argmax odd]*M - c*x4 - d*x9); features chosen by greedy search",
    15: "form pm7,pm8,M*[argmax in Q={5,9}] + P={4,5,8} + rest; Q,P greedy",
}
