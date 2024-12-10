# Statistical foundations of machine learning software
# Python equivalent of R package gbcode
# Author: G. Bontempi

# sum_rv.py
# Script: shows the relation between the variance
# of z and the variance of z.bar=(sum of N i.i.d. random variables)

import numpy as np
import matplotlib.pyplot as plt

# Clear all variables and close all figures
plt.close('all')

R = 10000  # number of realizations of each variable
N = 100  # number of summed variables

sdev = 1
mu = 1

z = np.random.normal(mu, sdev, R)  # z is normally distributed
print(f"Var[z] = {np.var(z)}")

plt.figure()
plt.hist(z, bins=50)
plt.title(f"single r.v.: mean={mu} variance={sdev**2}")
plt.show()

SN = np.zeros(R)
for n in range(N):
    SN += np.random.normal(mu, sdev, R)

print(f"Var[Sum(z)]/Var[z] = {np.var(SN)/np.var(z)}")

plt.figure()
plt.hist(SN, bins=50)
plt.title(f"Sum of {N} r.v.s (mu={mu},var={sdev**2}): variance ~{round(np.var(SN))}")
plt.show()

zbar = SN / N
plt.figure()
plt.hist(zbar, bins=50)
plt.title(f"Average of {N} r.v.s (mu={mu},var={sdev**2}): variance ~{round(np.var(zbar), 4)}")
plt.show()


