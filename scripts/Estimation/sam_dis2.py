import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

# Visualization of the distribution of the estimator
# of the variance of a Gaussian random variable

N = 10
mu = 0
sdev = 10
R = 10000

I = np.arange(-50, 50, 0.5)
p = norm.pdf(I, loc=mu, scale=sdev)
plt.figure()
plt.plot(I, p)
plt.title(f"Distribution of r.v. z: var={sdev**2}")
plt.show()

var_hat = np.zeros((R, 1))
std_hat = np.zeros((R, 1))
for r in range(R):
    D = np.random.normal(loc=mu, scale=sdev, size=N)
    var_hat[r, 0] = np.var(D, ddof=1)
    std_hat[r, 0] = np.std(D, ddof=1)

I2 = np.arange(0, 2*sdev**2, 0.5)
plt.figure()
plt.hist(var_hat, density=True, bins='auto')
plt.title(f"Variance estimator on N={N} samples: mean={np.mean(var_hat):.2f}")
plt.show()

ch = (var_hat * (N-1)) / (sdev**2)
plt.figure()
plt.hist(ch, density=True, bins='auto')
p_var_hat = chi2.pdf(I2, df=N-1)
plt.plot(I2, p_var_hat)
plt.show()

