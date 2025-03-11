# "INFOF422 Statistical foundations of machine learning" course
# Python  package gbcodepy
# Author: G. Bontempi

import numpy as np
import matplotlib.pyplot as plt

mu_p = 1
sd_p = 1

mu_n = -1
sd_n = 1

TT = np.arange(-10, 10.01, 0.01)
FPR = np.zeros(len(TT))
SE = np.zeros(len(TT))
PR = np.zeros(len(TT))
AL = np.zeros(len(TT))
N = 2000
DNp = np.random.normal(mu_p, sd_p, N//2)
DNn = np.random.normal(mu_n, sd_n, N//2)

for tt in range(len(TT)):
    thr = TT[tt]
    
    FN = np.sum(DNp < thr)
    FP = np.sum(DNn > thr)
    TN = np.sum(DNn < thr)
    TP = np.sum(DNp > thr)
    FPR[tt] = FP / (FP + TN)
    SE[tt] = TP / (TP + FN)
    PR[tt] = TP / (TP + FP)
    AL[tt] = (TP + FP) / (N)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(FPR, SE, color="red", linestyle="-")
plt.plot(FPR, FPR)
plt.title("ROC curve")
plt.ylabel("SE (TPR)")

plt.subplot(1, 3, 2)
plt.plot(SE, PR, color="red", linestyle="-")
plt.title("PR curve")
plt.xlabel("SE (TPR)")

plt.subplot(1, 3, 3)
plt.plot(AL, SE, color="red", linestyle="-")
plt.plot(AL, AL)
plt.title("Lift curve")
plt.ylabel("SE (TPR)")
plt.xlabel("% alerts")

plt.tight_layout()
plt.show()
