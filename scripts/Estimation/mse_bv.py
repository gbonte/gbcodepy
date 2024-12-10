## "Statistical foundations of machine learning" software
## package gbcodpy
## Author: G. Bontempi


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

N = 10
mu = 0
sdev = 10
R = 10000

I = np.arange(-50, 50.5, 0.5)
p = norm.pdf(I, loc=mu, scale=sdev)
plt.plot(I, p)
plt.title(f"Distribution of r.v. z: var={sdev**2}")
plt.show()

mu_hat = np.zeros((R, 1))
for r in range(R):
    D = np.random.normal(loc=mu, scale=sdev, size=N)
    mu_hat[r, 0] = np.mean(D)

err = mu_hat - mu

MSE = np.mean(err**2)
BIAS = np.mean(mu_hat) - mu
VARIANCE = np.var(mu_hat)

print(f"MSE={MSE:.2f}")
print(f"BIAS^2+VARIANCE={BIAS**2 + VARIANCE:.2f}")

