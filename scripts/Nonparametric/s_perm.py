import math
import random
import numpy as np
import matplotlib.pyplot as plt

# In R: rm(list=ls())
# In Python, explicit environment clearing is not required.

def fact(n):
    f = 1
    for i in range(2, n + 1):
        f = f * i
    return f

D1 = [74, 86, 98, 102]
D2 = [10, 25, 80]
M = len(D1)
N = len(D2)
D = D1 + D2

t = (sum(D[0:M]) / M) - (sum(D[M:M + N]) / N)

F = fact(M + N)
tp = np.zeros((F, 1))  # vecteur des résultats.
for p in range(F):
    Dp = random.sample(D, len(D))  # sample returns a random permutation of D
    tp[p, 0] = (sum(Dp[0:M]) / M) - (sum(Dp[M:M + N]) / N)

tp = np.sort(tp, axis=0)

q_inf = np.sum(t < tp) / tp.size
q_sup = np.sum(t > tp) / tp.size

plt.hist(tp, bins=30, edgecolor='black')
plt.axvline(x=t, color="red")

alpha = 0.1
if (q_inf < alpha / 2) or (q_sup < alpha / 2):
    plt.title("Hypothesis D1=D2 rejected: p-value=" + str(round(min(q_inf, q_sup),2)) + " alpha=" + str(alpha))
else:
    plt.title("Hypothesis D1=D2 not rejected: p-value=" + str(round(min(q_inf, q_sup),2)) + " alpha=" + str(alpha))

plt.show()
