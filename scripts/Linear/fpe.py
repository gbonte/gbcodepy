# "INFOF422 Statistical foundations of machine learning" course
## fpe.py
# Python translation of the R package gbcode 
# Author: G. Bontempi

from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets
import matplotlib
import time 

np.random.seed(0)

n = 3  # number of input variables
p = n + 1
p_max = 25
N = 35  # number of training data
x = np.sort(np.random.uniform(-1, 1, N))

X = np.ones((N, 1))
for j in range(1, p_max + 1):
    X = np.hstack((X, x.reshape(-1, 1) ** j))

xts = np.arange(-1, 1.01, 0.01)
Xts = np.ones((len(xts), 1))
for j in range(1, p_max + 1):
    Xts = np.hstack((Xts, xts.reshape(-1, 1) ** j))

beta = np.hstack(([1], np.arange(1, n + 1))).reshape(-1, 1)

sd_w = 0.5

f = X[:, :p] @ beta
Y = f.flatten() + np.random.normal(0, sd_w, N)

fts = Xts[:, :p] @ beta
Yts = fts.flatten() + np.random.normal(0, sd_w, len(fts))

R_emp = []
MISE = []
FPE = []
PSE = []
no_par = []
plt.figure()
for i in range(2, min(p_max, N - 1) + 1):
    
    XX = X[:, :i]
    invX = pinv(XX.T @ XX)
    beta_hat = invX @ XX.T @ Y
    Y_hat = XX @ beta_hat

    XXts = Xts[:, :i]
    Y_hats = XXts @ beta_hat
    no_par.append(i)

    e = Y - Y_hat
    R_emp.append((e.T @ e) / N)
    sde2hat = (e.T @ e) / (N - i)
    
    fig, ax = plt.subplots(figsize=(12, 10))  # Create figure and axes

    ax.plot(xts, fts, label='True function', color='green', linewidth=3)
    ax.scatter(x, Y, label='Data Points', color='blue')
    ax.plot(xts, Y_hats, label='Fitted function', color='red')
    plt.ylim(min(Y), max(Y))
    ax.legend()
    
    e_ts = Yts - Y_hats
    MISE.append((e_ts.T @ e_ts) / N)
    FPE.append(((1 + i / N) / (1 - i / N)) * (e.T @ e) / N)
    PSE.append((e.T @ e) / N + 2 * sde2hat * i / N)
    
    plt.title(f"degree={i-1}; MISE_emp={R_emp[i-2]:.4f}; FPE={FPE[i-2]:.2f}; PSE={PSE[i-2]:.3f}")
    
    display(fig)
    clear_output(wait=True)  # Clear the output
    
    plt.close(fig)  # Close the figure
    time.sleep(0.1)
    input(" ")
    
    
    plt.ioff()  # Disable interactive mode

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].plot(np.array(no_par) - 1, R_emp, label='Empirical risk')
axs[0, 0].set_xlabel("# parameters")
axs[0, 0].set_ylabel("Empirical risk")
axs[0, 0].set_title("Empirical risk")
axs[0, 0].set_xlim(2, 10)
axs[0, 0].set_ylim(0.05, 0.4)

axs[0, 1].plot(np.array(no_par) - 1, MISE, label='Generalization error')
axs[0, 1].set_xlabel("# parameters")
axs[0, 1].set_ylabel("Generalization error")
axs[0, 1].set_title("Generalization error")
axs[0, 1].set_xlim(2, 10)
axs[0, 1].set_ylim(1, 4)

axs[1, 0].plot(np.array(no_par) - 1, FPE, label='FPE')
axs[1, 0].set_xlabel("# parameters")
axs[1, 0].set_ylabel("FPE")
axs[1, 0].set_title("FPE")
axs[1, 0].set_xlim(2, 10)
axs[1, 0].set_ylim(0.05, 0.5)

axs[1, 1].plot(np.array(no_par) - 1, PSE, label='PSE')
axs[1, 1].set_xlabel("# parameters")
axs[1, 1].set_ylabel("PSE")
axs[1, 1].set_title("PSE")
axs[1, 1].set_xlim(2, 10)
axs[1, 1].set_ylim(0.05, 0.5)

plt.tight_layout()
plt.show()
plt.close()

print(f"which.min(R.emp)={np.argmin(R_emp) + 1}")
print(f"which.min(MISE)={np.argmin(MISE) + 1}")
print(f"which.min(FPE)={np.argmin(FPE) + 1}")
print(f"which.min(PSE)={np.argmin(PSE) + 1}")
