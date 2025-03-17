# "INFOF422 Statistical foundations of machine learning" course
# Python package gbcodepy
# Author: G. Bontempi
# Visualisation of the sampling distribution of the least square prediction by Monte Carlo simulation
#################################################################
# bvis.py                                                     #




import numpy as np
import matplotlib.pyplot as plt

R = 500 ## number of MC trials
N = 30 ## number of observations
beta0 = 0   ## intercept of the linear data generating process
beta1 = 1   ## slope of the linear data generating process

X = np.sort(np.random.normal(0, 1, N))

Yhatr = None

for r in range(R):
    
    ## dataset generation of size N
    Y = beta0 + beta1 * X + np.random.normal(0, 0.5, N)   
    XX = np.column_stack((np.ones(N), X))

    ## Least squares 
    betahat = np.linalg.solve(XX.T @ XX, XX.T @ Y)
    
    prediction = betahat[0] + betahat[1] * X
    if Yhatr is None:
        Yhatr = prediction.reshape(-1, 1)
        plt.figure()
    else:
        Yhatr = np.column_stack((prediction, Yhatr))
        
    plt.plot(X, prediction, color='black')
   

expected_prediction = np.mean(Yhatr, axis=1)

plt.plot(X, expected_prediction, 'g-', label='Expected Prediction')
plt.legend()
plt.show()
