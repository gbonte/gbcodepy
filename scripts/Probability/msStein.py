# "Statistical foundations of machine learning" software
# Python implementation
# Original Author: G. Bontempi
# Monte Carlo validation of Stein's lemma
# E[(z−mu)g(z)] = sigma^2 E[g′(z)]
# where g is a differentiable function and Z is Gaussian

import numpy as np

def g(z):
    return {
        'g': 3 * z**3,
        'dg': 9 * z**2
    }

n = 13
R = 5000
sdw = 0.3
mu = 1

P = []
D = []

for r in range(R):
    z = np.random.normal(loc=mu, scale=sdw, size=n)
    G = g(z)
    P.append(np.dot(z - mu, G['g']))
    D.append(sdw**2 * np.sum(G['dg']))

print(f"\n MC left-hand term= {np.mean(P)}")
print(f"\n MC right-hand term= {np.mean(D)}")
