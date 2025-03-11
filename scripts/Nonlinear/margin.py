import numpy as np
import matplotlib.pyplot as plt

N = 1000
# Y is sampled from the vector [-1, 1] with replacement (equivalent to sample(c(-1,1), N, rep=TRUE))
Y = np.random.choice([-1, 1], size=N, replace=True)
# h is sampled from a uniform distribution between -4 and 4 (equivalent to runif(N, -4, 4))
h = np.random.uniform(-3, 3, N)

M = h * Y
# iM holds the indices that would sort M in ascending order (equivalent to sort(M, decreasing=FALSE, index.return=TRUE)$ix)
iM = np.argsort(M)

# Plotting (equivalent to plot(M[iM], (h[iM]-Y[iM])^2, col="red", type="l", xlab="Margin", ylab="losses"))
plt.plot(M[iM], (h[iM] - Y[iM])**2, color="red", linestyle="-")
plt.xlabel("Margin")
plt.ylabel("losses")

# Adding lines (equivalent to lines(M[iM], exp(-M[iM])))
plt.plot(M[iM], np.exp(-M[iM]))

plt.show()
