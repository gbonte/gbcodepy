import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(0)

mu = 10
sigma = 0.1
N = 100
alpha = 0.1
z_alpha = norm.ppf(1 - alpha/2)

seq_N_iter = np.arange(100, 10000, 25)
perc = []

for N_iter in seq_N_iter:
    ins = np.zeros((N_iter, 1))
    for i in range(N_iter):
        D = np.random.normal(mu, sigma, N)
        mu_hat = np.mean(D)
        ins[i, 0] = (mu_hat > (mu - z_alpha * sigma / np.sqrt(N))) & (mu_hat < (mu + z_alpha * sigma / np.sqrt(N)))
    
    perc.append(np.sum(ins) / N_iter)

one = np.full(len(perc), 1 - alpha)
Nit = len(seq_N_iter)

plt.figure(figsize=(10, 6))
plt.plot(seq_N_iter, one, label='1 - alpha')
plt.plot(seq_N_iter, perc, label='Observed frequency')
plt.ylim(0.85, 1)
plt.xlabel('Number of iterations')
plt.ylabel('Frequency of event "relation satisfied"')
plt.title(f'alpha = {alpha}, N = {N}')
plt.legend()
plt.show()

