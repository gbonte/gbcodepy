import numpy as np
import matplotlib.pyplot as plt

def dopler(x):
    return 20 * np.sqrt(x * (1 - x)) * np.sin(2 * np.pi * 1.05 / (x + 0.05))

def dataset_dopler(N, sigma=1):
    np.random.seed(0)
    x = np.sort(np.random.uniform(0.12, 1, N))
    y = dopler(x) + np.random.normal(scale=sigma, size=N)
    x_ts = np.sort(np.random.uniform(0.12, 1, N))
    y_ts = dopler(x_ts)
    return {'x': x, 'y': y, 'x.ts': x_ts, 'y.ts': y_ts}

# Reset plotting parameters (equivalent to par(mfrow=c(1,1)) in R)
plt.figure()

D = dataset_dopler(100, 0.1)
plt.plot(D['x'], D['y'], 'o')
plt.show()
