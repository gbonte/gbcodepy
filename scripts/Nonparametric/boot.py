#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# "INFOF422 Statistical foundations of machine learning" course
# Python version of gbcode package
# boot.py
# Bootstrap estimation of the 90% quantile of a uniform distribution between 0 and 10
# The simulation shows the impact of the number of observations (estimation error) and of the number of bootstrap replications (simulation error)
# Author: G. Bontempi

import numpy as np
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

np.random.seed(0)

lower = 0
upper = 10
p = 0.9
theta = lower + p * (upper - lower)

Qb = []
for N in range(250, 5001, 100):
    Thetahat = []
    DN = np.random.uniform(lower, upper, N)
    sB = range(10, 1501, 100)
    for B in sB:
        for b in range(B):
            Db = np.random.choice(DN, size=N, replace=True)
            Qb.append(np.quantile(Db, p))
        #   plt.hist(Qb)
        #   plt.title(f"# obs.={N} # bootstrap samples={B}")
        thetahat = np.mean(Qb)
        Thetahat.append(thetahat)
        # print(f"# obs.={N}, # bootstrap samples={B}, estim={thetahat}, %err={abs((theta - thetahat)/theta)}")
    
    fig, ax = plt.subplots(figsize=(8, 8)) 
    ax.plot(sB, Thetahat)
    ax.axhline(y=theta, color='r')
    plt.title(f"Theta={theta} Number obs={N}")
    plt.xlabel("Number of bootstrap repetitions")
    plt.ylim(theta * 0.95, theta * 1.05)
    
    display(fig)
    #clear_output(wait=True)  # Clear the output
    
    plt.close(fig)  # Close the figure
    time.sleep(1)
    #input()
    #plt.show()
