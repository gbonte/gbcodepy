import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Clear the workspace (not typically done in Python; variables are re-assigned as needed)
# In Python, we generally don't clear the namespace in a script

## distribution of maximum of N normal variables 
## pN(z)=N[pnorm(z)]^(N-1)*dnorm(z)

## distribution of minimum of N normal variables 
## p1(z)=N[1-pnorm(z)]^(N-1)*dnorm(z)

muz = 2
sdz = 2
seqN = np.arange(5, 101, 5)
Z = np.linspace(muz - 5 * sdz, muz + 5 * sdz, 500)

# Plot the standard normal density curve for given mean and sd
plt.plot(Z, norm.pdf(Z, loc=muz, scale=sdz), 
         linestyle='-', color='blue', linewidth=3, 
         label="", zorder=1)
plt.ylim(0, 1)
plt.title("N=" + str(np.max(seqN)) + " i.i.d. r.v.s. zi.  Density min(zi) (green) and max(zi) (red)")

for N in seqN:
    pN = []
    for z in Z:
        # Compute the density of the maximum of N normal variables at point z
        pN.append(N * (norm.cdf(z, loc=muz, scale=sdz) ** (N - 1)) * norm.pdf(z, loc=muz, scale=sdz))
    plt.plot(Z, pN, color="red")
    
    p1 = []
    for z in Z:
        # Compute the density of the minimum of N normal variables at point z
        p1.append(N * ((1 - norm.cdf(z, loc=muz, scale=sdz)) ** (N - 1)) * norm.pdf(z, loc=muz, scale=sdz))
    plt.plot(Z, p1, color="green")

plt.show()
