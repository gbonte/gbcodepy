## "Statistical foundations of machine learning" software
## Python package gbcodepy
## Author: G. Bontempi

# cumdis_2.py
# Script: visualizes the unbiasedness of the empirical distribution function


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Set interactive mode similar to par(ask=TRUE) in R so that the plot window waits for a key press
plt.ion()

# rm(list=ls()): In Python, we reinitialize variables instead of clearing the workspace.
N = 10
R = 50
I = np.arange(-5, 5.1, 0.1)  # equivalent to seq(-5,5,by=.1); ensure 5 is included
emp = np.empty((0, len(I)))  # Initialize an empty array to accumulate rows, similar to emp <- NULL

# Define a function to compute the empirical cumulative distribution function (ECDF)
def ecdf(x, data):
    # For each value xi in x, compute the proportion of data points in 'data' that are less than or equal to xi
    return np.array([np.sum(data <= xi) / len(data) for xi in x])

for i in range(1, R + 1):
    # Generate N samples from a standard normal distribution; equivalent to DN <- rnorm(N)
    DN = np.random.normal(0, 1, N)
    
    # Create the empirical cumulative distribution function for DN
    F = lambda x: ecdf(x, DN)  # Equivalent to F <- ecdf(DN)
    
    # Append the ECDF evaluated at I to 'emp'; equivalent to emp <- rbind(emp, F(I))
    F_I = F(I)
    emp = np.vstack([emp, F_I])
    
    # Compute the average of the empirical function across all rows; equivalent to m.emp <- apply(emp, 2, mean)
    m_emp = np.mean(emp, axis=0)
    
    # Plot the average empirical distribution function
    plt.figure()
    plt.plot(I, m_emp, label="Empirical", linestyle='--')  # Dashed line similar to lty=3
    # Plot the theoretical Gaussian distribution function; equivalent to lines(I, pnorm(I), pch=15)
    plt.plot(I, norm.cdf(I), label="Gaussian", linestyle='-')  # Solid line similar to lty=1; marker pch=15 is ignored
    plt.title("Average of {} empirical distributions made with {} samples".format(i, N))
    plt.legend(loc='upper left')
    plt.xlabel("I")
    plt.ylabel("Value")
    plt.show()
    plt.pause(0.1)  # Pause to update the plot; mimicking interactive behavior
    plt.close()

