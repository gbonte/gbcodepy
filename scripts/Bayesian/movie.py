
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


plt.ion()

n = 10
pr_theta = [0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0, 0]  # A priori distribution
po_theta = np.zeros(n)  # A pposteriori sistribution.
DN = [6, 7, 8, 9]  # Sample.
var = 4  # Known Variance .

def lik(m, D, var):
    # empirical likelihood function
    N = len(D)
    Lik = 1
    for i in range(N):  
        Lik = Lik * norm.pdf(D[i], loc=m, scale=math.sqrt(var))
    return Lik

# Computation of a posteriori distribution
# =======================================
marg = 0
for theta in range(1, n+1):
    marg = marg + lik(theta, DN, var) * pr_theta[theta-1]

for theta in range(1, n+1):
    po_theta[theta-1] = (lik(theta, DN, var) * pr_theta[theta-1]) / marg

# Distributions plot
# =====================================================
x_values = list(range(1, n+1))
plt.figure()
plt.plot(x_values, pr_theta, marker='o', linestyle='-', color="red")
plt.ylim(0, 0.4)
plt.title('Distribution of theta')
plt.plot(x_values, po_theta, marker='o', linestyle='-', color="blue")
ax = plt.gca()
ax.legend(["Prior", "Posterior"], loc="upper left", bbox_to_anchor=(2, 0.4))
plt.show()


