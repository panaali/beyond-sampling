import numpy as np, probe
np.set_printoptions(precision=1,suppress=True,linewidth=170)
E=np.eye(10)
print('positive impulse at j (rows j, cols neuron)');print(probe.hidden(E))
print('negative impulse');print(probe.hidden(-E))
# baseline: all equal -1 except impulse? background of small noise
