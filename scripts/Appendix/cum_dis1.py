# ## "INFOF422 Statistical foundations of machine learning" course
# ## R package gbcode 
# ## Author: G. Bontempi

# # cumdis_1.py
# # Script: visualizes the unbiasedness of the empirical distribution function 


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def ecdf(data):
    # This function mimics R's ecdf, returning a function which, for given x,
    # returns the proportion of data less than or equal to each element of x.
    sorted_data = np.sort(data)
    n = len(sorted_data)
    def F(x):
        # Using searchsorted for efficient vectorized computation in sorted_data.
        return np.searchsorted(sorted_data, x, side='right') / n
    return F


N = 50

I = np.arange(-5, 5.01, 0.1)
emp = []  # Initialize emp as an empty list to store rows of empirical CDF values.
DN = np.random.randn(3)

# Loop from n=4 to n=N (inclusive)
for n in range(4, N + 1):
    # Extend DN with 10 new random normal observations
    DN = np.concatenate([DN, np.random.randn(10)])
    F = ecdf(DN)
    # Append the current empirical CDF values evaluated at I as a new row in emp
    emp.append(F(I))
    # Compute the average of the empirical function across all rows (column-wise mean)
    m_emp = np.mean(np.array(emp), axis=0)  # average of the empirical function 
    
    # Create a new figure for each iteration
    plt.figure()
    # Plot the average empirical distribution
    plt.plot(I, m_emp, linestyle=':', label="Empirical")
    # Set the plot title with the current size of DN
    plt.plot(I, norm.cdf(I), label="Gaussian", linestyle='-')  # Solid line similar to lty=1; marker pch=15 is ignored

    plt.title(" Empirical distributions, N=" + str(len(DN)))
    plt.pause(0.1) 
    # Overlay the Gaussian distribution function using the standard normal CDF, with square markers (pch=15 equivalent)
