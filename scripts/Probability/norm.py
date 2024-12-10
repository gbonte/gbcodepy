## "INFOF422 Statistical foundations of machine learning" course
## package gbcode 
## Author: G. Bontempi

import numpy as np
from scipy import stats

np.random.seed(0)
N = 1000000
mu = 1
sigma = 2

DN = np.random.normal(mu, sigma, N)

for size in [1, 1.282, 1.645, 1.96, 2, 2.57, 3]:
    monte_carlo = np.sum((DN <= (mu + size * sigma)) & (DN >= (mu - size * sigma))) / N
    analytical = stats.norm.cdf(mu + size * sigma, mu, sigma) - stats.norm.cdf(mu - size * sigma, mu, sigma)
    print(f"P[mu-{size}sigma <= z <= mu+{size}sigma] Monte Carlo: {monte_carlo:.6f} analytical: {analytical:.6f}")

