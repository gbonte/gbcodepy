import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Clear all variables (equivalent to rm(list=ls()) in R)
# In Python, we don't need to clear variables explicitly

N = 11
z = 7
p = np.arange(0.01, 0.99, 0.01)

# Set up interactive mode for plots
plt.ion()

# Likelihood plot
L = [binom.pmf(z, N, pi) for pi in p]
plt.figure()
plt.plot(p, L)
plt.title("Likelihood")
plt.xlabel("Probability p")
plt.ylabel("L(p)")
plt.axvline(x=z/N, linestyle='--')
plt.show()

# Log-Likelihood plot
plt.figure()
plt.plot(p, np.log(L))
plt.title("Log-Likelihood")
plt.xlabel("Probability p")
plt.ylabel("l(p)")
plt.axvline(x=z/N, linestyle='--')
plt.show()

# Keep plots open until user closes them
plt.ioff()
plt.show()

