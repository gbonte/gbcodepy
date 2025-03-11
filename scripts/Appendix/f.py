import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

# "Statistical foundations of machine learning" software
# package gbcode
# Author: G. Bontempi

# script s_f.R
N1 = 10
N2 = 20
x = np.arange(-0.1, 5.0 + 0.1, 0.1)
plt.figure()
plt.plot(x, f.pdf(x, N1, N2), linestyle='-')
plt.title("F (N1=" + str(N1) + ",N2=" + str(N2) + ") density")
plt.xlabel("x")
plt.ylabel("Density")
plt.show()

plt.figure()
plt.plot(x, f.cdf(x, N1, N2), linestyle='-')
plt.title("F (N1=" + str(N1) + ",N2=" + str(N2) + ") cumulative distribution")
plt.xlabel("x")
plt.ylabel("Cumulative Distribution")
plt.show()

