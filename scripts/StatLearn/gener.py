## # "Statistical foundations of machine learning" software
# Author: G. Bontempi
# gener.py
# # Monte Carlo estimation of generalization error




import numpy as np
import matplotlib.pyplot as plt

S = 5000       # number of MC trials
sdw = 1        # noise stdev
N = 50         # number of training examples

range_alpha = np.arange(1.4, 3.4 + 0.005, 0.01)  # set Lambda of alpha (inclusive on 3.4)

RalphaN = []    # storage for test risk with chosen alpha per trial
Ralpha = []     # will store test risk computed for each alpha in range_alpha later
Remp = []       # storage for in-sample empirical risk for chosen alpha each trial
RempDN = np.full((S, len(range_alpha)), np.nan)  # empirical risks over the grid for each trial
AL = []         # storage for selected alpha per trial

# Test set for functional risk computation
Nts = 100000
Xts = np.random.uniform(-2, 2, Nts)
Yts = Xts**3 + np.random.normal(0, sdw, Nts)

# Monte Carlo simulation
for s in range(S):
    np.random.seed(s+1)  # setting seed; using s+1 to mimic R's 1-indexing seed
    # Dataset generation
    Xtr = np.random.uniform(-2, 2, N)
    Ytr = Xtr**3 + np.random.normal(0, sdw, N)  # f(x)=x^3
    
    # ERM parametric identification
    bestRemp = np.inf
    cnt = 0
    for alpha in range_alpha:
        RempN = np.mean((Ytr - alpha * Xtr)**2)
        RempDN[s, cnt] = RempN
        if RempN < bestRemp:
            bestRemp = RempN
            alphaN = alpha
        cnt += 1
    
    Remp.append(bestRemp)
    RalphaN.append(np.mean((Yts - alphaN * Xts)**2))
    AL.append(alphaN)

# Compute R(alpha) for each alpha in range_alpha over the test set
for alpha in range_alpha:
    Ralpha.append(np.mean((Yts - alpha * Xts)**2))

# Print results similar to R's cat function
print("\n E[R(alpha_N)]=", np.mean(RalphaN))
print("\n alpha_0=", range_alpha[np.argmin(Ralpha)])

# Plotting
plt.figure(figsize=(16, 4))

# First subplot: histogram of R(alpha_N)
plt.subplot(1, 4, 1)
plt.hist(RalphaN, bins=30, edgecolor='black')
plt.title("Sampling distribution of R(alpha_N)")
plt.axvline(np.mean(RalphaN), color="green", label=f"Mean: {np.mean(RalphaN):.3f}")
plt.legend()

# Second subplot: histogram of Remp(alpha_N)
plt.subplot(1, 4, 2)
plt.hist(Remp, bins=30, edgecolor='black')
plt.title("Sampling distribution of Remp(alpha_N)")
plt.axvline(np.mean(Remp), color="red", label=f"Mean: {np.mean(Remp):.3f}")
plt.legend()

# Third subplot: histogram of alpha_N
plt.subplot(1, 4, 3)
plt.hist(AL, bins=30, edgecolor='black')
plt.title("Sampling distribution of alpha_N")
plt.xlabel("alpha_N")

# Fourth subplot: plot of R(alpha) vs alpha
plt.subplot(1, 4, 4)
plt.plot(range_alpha, Ralpha, lw=2)
plt.xlim(2, 3)
plt.ylim(2.4, 2.8)
plt.ylabel('R(alpha)')
plt.xlabel("alpha")
min_index = np.argmin(Ralpha)
plt.plot(range_alpha[min_index], Ralpha[min_index], 'o', markersize=8, label="Min R(alpha)")
plt.axhline(np.mean(Remp), color="red", lw=2, label="Mean Remp")
plt.axhline(np.mean(RalphaN), color="green", lw=2, label="Mean R(alpha_N)")
plt.legend()

plt.tight_layout()
plt.show()

# New plot similar to the last part of the R code:
plt.figure(figsize=(8, 6))
plt.plot(range_alpha, Ralpha, lw=3, label="R(alpha)")
plt.ylim(1.5, 3.8)
plt.xlabel("alpha")
plt.ylabel("R(alpha)")

# Analytical expression of R(alpha): 4*alpha^2/3 - (32/5)*alpha + 71/7
analytical_R = 4 * range_alpha**2 / 3 - (32/5) * range_alpha + 71/7
plt.plot(range_alpha, analytical_R, lw=3, color="orange", label="Analytical R(alpha)")

# Plotting dashed empirical risk curves for the first 50 samples
for s in range(50):
    plt.plot(range_alpha, RempDN[s, :], linestyle="dashed", color="gray")
    min_idx = np.argmin(RempDN[s, :])
    plt.plot(range_alpha[min_idx], 1.5, 'o', markersize=4, color="black")

plt.legend()
plt.show()
