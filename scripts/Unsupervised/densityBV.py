import numpy as np
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt

# Bias/variance analysis of density estimator

Gauss = False
muz = 0
sdz = 0.5
z = 0
if Gauss:
    pz = norm.pdf(z, loc=muz, scale=sdz)
    # Gaussian distribution
else:
    pz = uniform.pdf(z, loc=-sdz, scale=2*sdz)
    # Uniform distribution

N = 100
R = 30000
LB = 20
B = np.linspace(0.05, 1, num=LB)

phat = np.full((R, LB), np.nan)

for r in range(R):
    for j in range(LB):
        if Gauss:
            DN = np.random.normal(muz, sdz, N)
        else:
            DN = np.random.normal(-sdz, sdz, N)
        k = np.sum(np.abs(DN - z) < B[j])
        phat[r, j] = k / (N * 2 * B[j])  # density estimator

Bias = np.abs(np.mean(phat, axis=0) - pz)
Var = np.var(phat, axis=0)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(B, Bias, label="Bias")
plt.xlabel("V")
plt.title("Bias")

plt.subplot(1, 2, 2)
plt.plot(B, Var, label="Variance")
plt.xlabel("V")
plt.title("Variance")

plt.tight_layout()
plt.show()
