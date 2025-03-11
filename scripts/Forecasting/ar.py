# "Statistical foundations of machine learning" software
# package gbcodepy
# Author: G. Bontempi


import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

np.random.seed(0)

q = 4  ## order of AR(q)
N = q * 500

alpha = np.random.randn(q)

## stationarity check
while np.any(np.abs(np.roots(np.concatenate(([1], -alpha))) ) > 1):
    alpha = np.random.randn(q)

Y = list(np.random.randn(q))
for i in range(q, N):
    L = len(Y)
    # Take the last q values, reversed, and compute the dot product with alpha,
    # then add a random noise with standard deviation 0.1
    Y.append( np.dot(alpha, np.array(Y[L - q:])[::-1]) + np.random.normal(scale=0.1) )
Y = np.array(Y[q:])

N = len(Y)
Q = 20

hatacf = np.zeros(Q)
## Estimation auto-correlation function

for k in range(1, Q+1):
    # Create a vector Yk by prepending k zeros to Y
    Yk = np.concatenate((np.zeros(k), Y))
    # Compute correlation between Y[(k+1):N] and Yk[(k+1):N]
    # In R, indices are 1-indexed; here we adjust to 0-indexing.
    Y_segment = Y[k:N]
    Yk_segment = Yk[k:N]
    # Calculate Pearson correlation coefficient
    if np.std(Y_segment) != 0 and np.std(Yk_segment) != 0:
        C = np.corrcoef(Y_segment, Yk_segment)[0, 1]
    else:
        C = 0.0
    hatacf[k-1] = C

YY = Y.reshape((N, 1))
# For each k from 1 to Q+1, cbind a new column to YY:
for k in range(1, Q+2):
    # Create a column vector: first (N-k) entries from Y[(k+1):N] and then k NaN values.
    newcol = np.concatenate((Y[k:N], np.full(k, np.nan)))
    YY = np.column_stack((YY, newcol))

## Estimation partial auto-correlation function by linear regression
hatpacf = []
for k in range(2, Q+2):
    # Prepare the design matrix using columns 2 to k of YY (1-indexed in R corresponds to index 1:k in Python)
    X = YY[:, 1:k]
    y_dep = YY[:, 0]
    # Remove rows with NaN values
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y_dep)
    X_valid = X[valid, :]
    y_valid = y_dep[valid]
    # Add constant term for intercept
    X_const = sm.add_constant(X_valid, has_constant='add')
    model = sm.OLS(y_valid, X_const)
    results = model.fit()
    # In R, lm(YY[,1] ~ YY[,2:k])$coefficients[k] corresponds to the coefficient of the last predictor.
    hatpacf.append(results.params[-1])

## Setup plotting: 3 plots in one column
plt.figure(figsize=(6, 12))
plt.subplots_adjust(hspace=0.5)

# Plot 1: Time series Y with title "AR(q)"
plt.subplot(3, 1, 1)
plt.plot(Y, linestyle='-', marker='o', markersize=2)
plt.title("AR({})".format(q))
plt.xlabel('')
plt.grid(True)

# Plot 2: Estimated acf
plt.subplot(3, 1, 2)
plt.plot(np.arange(1, Q+1), hatacf, linestyle='-', marker='o')
plt.xlabel('k')
plt.ylabel('')
plt.title("Est acf")
# Add a horizontal line at threshold 2/sqrt(N)
thr = 2/np.sqrt(N)
plt.hlines(thr, 1, Q, linestyles='dashed')
plt.legend(['Estimated autocor', 'thr'], loc='lower right', fontsize=8)
plt.grid(True)

# Plot 3: Estimated pacf
plt.subplot(3, 1, 3)
plt.plot(np.arange(1, Q+1), hatpacf, linestyle='-', marker='o')
plt.xlabel('k')
plt.ylabel('')
plt.title("Est pacf")
plt.hlines(thr, 1, Q, linestyles='dashed')
plt.legend(['Estimated pcor', 'thr'], loc='lower right', fontsize=8)
plt.grid(True)

plt.show()

# Print acf values (excluding the first value) for lags 1 to Q-2
acf_vals = acf(Y, nlags=Q, fft=False)
print(acf_vals[1:(Q-1)])
print("\n ")

# Print pacf values (excluding the first value) for lags 1 to Q-1
pacf_vals = pacf(Y, nlags=Q)
print(pacf_vals[0:(Q-1)])
print("\n ")

## Setup plotting: Plot time series, acf and pacf using statsmodels functions
plt.figure(figsize=(6, 12))
plt.subplots_adjust(hspace=0.5)

# Plot time series Y with title "AR(q)"
plt.subplot(3, 1, 1)
plt.plot(Y, linestyle='-', marker='o', markersize=2)
plt.title("AR({})".format(q))
plt.xlabel('')
plt.grid(True)

# Plot acf using statsmodels' plot_acf
plt.subplot(3, 1, 2)
plot_acf(Y, ax=plt.gca())
plt.grid(True)

# Plot pacf using statsmodels' plot_pacf
plt.subplot(3, 1, 3)
plot_pacf(Y, ax=plt.gca(), method='ywm')
plt.grid(True)

plt.show()
