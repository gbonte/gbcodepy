# "Statistical foundations of machine learning" software
# Python package gbcodepy 
# Author: G. Bontempi
# script stu.py

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

N = 10
x = np.arange(-5, 5.1, 0.1)
plt.figure()
plt.plot(x, stats.t.pdf(x, df=N))
plt.title("Student (N=" + str(N) + ") density")
plt.show()

plt.figure()
plt.plot(x, stats.t.cdf(x, df=N))
plt.title("Student (N=" + str(N) + ") cumulative distribution")
plt.show()
