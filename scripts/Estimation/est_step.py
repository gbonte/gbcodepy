## "Statistical foundations of machine learning" software
## package gbcodpy
## Author: G. Bontempi

# est_step.R
# Script: visualizes the distributions of 4 estimators
# 1: sample average as estimator of the mean 
# 2: sample stdev as estimator of stdev
# 3: sample min as (biased) estimator  of the mean
# 4: sample width as (biased) estimator of sdev

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gaussian_kde

R = 10000

muz = 1
sdz = 1
N = 20
muhat = []
sdhat = []
muhat2 = []
sdhat2 = []

for r in range(1, R+1):
    DN = np.random.normal(muz, sdz, N)
    xaxis = np.arange(muz-3*sdz, muz+3*sdz, 0.01)
    
    plt.figure(figsize=(15, 15))
    plt.subplot(3, 2, 1)
    plt.plot(DN, np.zeros_like(DN), 'o')
    plt.title(f"mu={muz} sd={sdz} N={N}")
    plt.xlabel("Dataset")
    plt.ylabel("")
    
    plt.subplot(3, 2, 2)
    plt.plot(xaxis, norm.pdf(xaxis, muz, sdz))
    plt.title("z density")
    plt.xlabel("z")
    
    muhat.append(np.mean(DN))
    sdhat.append(np.std(DN, ddof=1))
    muhat2.append(np.min(DN))
    sdhat2.append(0.5*(np.max(DN)-np.min(DN)))
    
    if r % 10 == 0:
        br = r // 15 if r > 30 else r // 2
        
        plt.subplot(3, 2, 3)
        plt.hist(muhat, bins=br, density=True)
        plt.title(f"Bias={np.mean(muhat)-muz:.2f} Var={np.var(muhat):.2f} MSE={np.mean((np.array(muhat)-muz)**2):.2f}")
        plt.xlabel("Sample average distribution")
        kde = gaussian_kde(muhat)
        plt.plot(xaxis, kde(xaxis), linewidth=2, color='chocolate')
        plt.plot(muhat, np.zeros_like(muhat), 'o')
        
        plt.subplot(3, 2, 4)
        plt.hist(sdhat, bins=br, density=True)
        plt.title(f"Bias={np.mean(sdhat)-sdz:.2f} Var={np.var(sdhat):.2f} MSE={np.mean((np.array(sdhat)-sdz)**2):.2f}")
        plt.xlabel("Sample stdev distribution")
        kde = gaussian_kde(sdhat)
        plt.plot(xaxis, kde(xaxis), linewidth=2, color='chocolate')
        plt.plot(sdhat, np.zeros_like(sdhat), 'o')
        
        plt.subplot(3, 2, 5)
        plt.hist(muhat2, bins=br, density=True)
        plt.title(f"Bias={np.mean(muhat2)-muz:.2f} Var={np.var(muhat2):.2f} MSE={np.mean((np.array(muhat2)-muz)**2):.2f}")
        plt.xlabel("Sample min distribution")
        kde = gaussian_kde(muhat2)
        plt.plot(xaxis, kde(xaxis), linewidth=2, color='chocolate')
        plt.plot(muhat2, np.zeros_like(muhat2), 'o')
        
        plt.subplot(3, 2, 6)
        plt.hist(sdhat2, bins=br, density=True)
        plt.title(f"Bias={np.mean(sdhat2)-sdz:.2f} Var={np.var(sdhat2):.2f} MSE={np.mean((np.array(sdhat2)-sdz)**2):.2f}")
        plt.xlabel("Sample width distribution")
        kde = gaussian_kde(sdhat2)
        plt.plot(xaxis, kde(xaxis), linewidth=2, color='chocolate')
        plt.plot(sdhat2, np.zeros_like(sdhat2), 'o')
        
        plt.tight_layout()
        plt.show()
        
        input(f"r={r} Make more runs...")

