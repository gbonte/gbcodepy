# ## "INFOF422 Statistical foundations of machine learning" course
# ## Python package gbcodepy
# ## Author: G. Bontempi
#
# ## script chisq.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

N = 10
x = np.arange(0, 50.1, 0.1)
plt.figure()
plt.plot(x, chi2.pdf(x, N))
plt.title("chi-squared (N=" + str(N) + ") density")

plt.figure()
plt.plot(x, chi2.cdf(x, N))
plt.title("chi-squared (N=" + str(N) + ") cumulative distribution")
plt.show()
