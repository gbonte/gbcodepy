import numpy as np
from scipy.stats import norm

# "INFOF422 Statistical foundations of machine learning" course
# Python translation of R package gbcode 
# Author: G. Bontempi
# mc_condentr.py
# This script uses Monte Carlo to compute entropy, conditional entropy and mutual information  
# and compare such approximations to the analytical values (Gaussian case)

r = 1000
r2 = 500
sdw = 0.1

def f(x1, x2):
    return np.sin(x1) * x2

# Y = x1 + x2 + w
# X1 ~ U(-1,1), X2 ~ U(-1,1) W ~ N(0, sdw^2)
# p(Y|x1, x2) = N(y - x1 - x2, 0, 0.1^2)

# H(Y|x1)
# p(Y|x1) = ∫x2 p(Y|x1, x2)
hy1 = []
for i in range(1, r + 1):
    np.random.seed(i)
    x1 = np.random.uniform(-1, 1)
    x2 = np.random.uniform(-1, 1)
    y = f(x1, x2) + np.random.normal(0, sdw)
    pY1 = []
    for j in range(1, r2 + 1):
        x2_new = np.random.uniform(-1, 1)
        pY1.append(norm.pdf(y - f(x1, x2_new), 0, sdw))
    hy1.append(-np.log(np.mean(pY1)))

# H(Y|x2)
# p(Y|x2) = ∫x1 p(Y|x1, x2)
hy2 = []
for i in range(1, r + 1):
    np.random.seed(i)
    x1 = np.random.uniform(-1, 1)
    x2 = np.random.uniform(-1, 1)
    y = f(x1, x2) + np.random.normal(0, sdw)
    pY2 = []
    for j in range(1, r2 + 1):
        x1_new = np.random.uniform(-1, 1)
        pY2.append(norm.pdf(y - f(x1_new, x2), 0, sdw))
    hy2.append(-np.log(np.mean(pY2)))

# H(Y|x1, x2)
# p(Y|x1, x2)
hy12 = []
for i in range(1, r + 1):
    np.random.seed(i)
    x1 = np.random.uniform(-1, 1)
    x2 = np.random.uniform(-1, 1)
    y = f(x1, x2) + np.random.normal(0, sdw)
    hy12.append(-norm.logpdf(y - f(x1, x2), 0, sdw))

# H(W)
hw = []
for i in range(1, r + 1):
    np.random.seed(i)
    w = np.random.normal(0, sdw)
    hw.append(-norm.logpdf(w, 0, sdw))

# H(Y)
hy = []
for i in range(1, r + 1):
    np.random.seed(i)
    w = np.random.normal(0, sdw)
    x1 = np.random.uniform(-1, 1)
    x2 = np.random.uniform(-1, 1)
    y = f(x1, x2) + np.random.normal(0, sdw)
    pY = []
    for j in range(1, r2 + 1):
        x2_new = np.random.uniform(-1, 1)
        x1_new = np.random.uniform(-1, 1)
        pY.append(norm.pdf(y - f(x1_new, x2_new), 0, sdw))
    hy.append(-np.log(np.mean(pY)))

print("\n --- \n MC H(Y|X1)=", np.mean(hy1),
      "MC H(Y|X2)=", np.mean(hy2),
      "MC H(Y|X1,X2)=", np.mean(hy12),
      " \n MC H(W)=", np.mean(hw),
      "analytical H(W)=", 0.5*(1 + np.log(2 * np.pi * sdw**2)),
      " MC H(Y)=", np.mean(hy))

print("\n--\n MC I(Y;X1)=", np.mean(hy) - np.mean(hy1),
      " MC I(Y;X2)=", np.mean(hy) - np.mean(hy2),
      " MC I(Y;X1,X2)=", np.mean(hy) - np.mean(hy12),
      "analytical I(Y;X1,X2)=", np.mean(hy) - 0.5*(1 + np.log(2 * np.pi * sdw**2)),
      "\n")
