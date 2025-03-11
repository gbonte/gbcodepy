import numpy as np
import matplotlib.pyplot as plt

# h() polynomial functions of different orders 
def hlog(x, alpha, ord):
    if ord == 1:
        # return(np.dot(alpha, np.concatenate(([1], x.reshape(-1)))))
        return(np.dot(alpha, np.concatenate(([1], x))))
    if ord == 2:
        return(np.dot(alpha, np.concatenate(([1], x, x**2))))
    if ord == 3:
        return(np.dot(alpha, np.concatenate(([1], x, x**2, x**3))))

def predlog(X, alpha, ord=1):
    N = X.shape[0]
    P1 = []
    for i in range(N):
        hi = min(10, hlog(X[i, :], alpha, ord))
        P1.append(np.exp(hi) / (1 + np.exp(hi)))
    return np.round(np.array(P1))

np.random.seed(0)
### Dataset creation
n = 2
mu_0 = np.array([1, -1])
mu_1 = np.array([-1, 1])
sdw = 1.5
N = 200

X0 = []
for i in range(N):
    X0.append([np.random.normal(mu_0[0], sdw), np.random.normal(mu_0[1], sdw)])
X0 = np.array(X0)

X1 = []
for i in range(N):
    X1.append([np.random.normal(mu_1[0], sdw), np.random.normal(mu_1[1], sdw)])
X1 = np.array(X1)

X = np.vstack((X0, X1))
Y = np.concatenate((np.zeros(N), np.zeros(N) + 1))

I = np.random.permutation(2 * N)
X = X[I, :]
Y = Y[I]

Xtr = X[0:N, :]
Ytr = Y[0:N]
Xts = X[N:2*N, :]
Yts = Y[N:2*N]

###########
R_iter = 500
ord = 1
p = 1 + n  # number of parameters
alpha = np.random.randn(p)

# gradient-based search of optimal set of parameters
bestJ = float('inf')
eta = 0.1

# Gradient-based 
for r in range(R_iter):
    dJ = np.zeros(p)
    for i in range(N):
        hi = hlog(Xtr[i, :], alpha, ord=1)
        dJ[0] = dJ[0] - Ytr[i] + np.exp(hi) / (1 + np.exp(hi))
        for j in range(n):
            dJ[j+1] = dJ[j+1] - Ytr[i] * Xtr[i, j] + (np.exp(hi) * Xtr[i, j]) / (1 + np.exp(hi))
    # dJ/da0= -yi+e^hi/(1+e^hi)
    # dJ/daj= -yi*xij+e^hi*xij/(1+e^hi)
    
    alpha = alpha - eta * dJ / N
    
    Yhat = predlog(Xtr, alpha, ord=1)
    Miscl = np.sum(Yhat != Ytr) / N
    print(" Miscl error=", Miscl)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], s=60, marker='o', edgecolor='k', facecolor='none', label='All points', zorder=1)
I0 = np.where(Y == 0)[0]
plt.scatter(X[I0, 0], X[I0, 1], color="red", s=60, marker='o', label='Y == 0', zorder=2)

Xrange = np.arange(-7, 7.1, 0.1)

x1_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
x2_vals = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
for x1 in x1_vals:
    for x2 in x2_vals:
        # predlog expects a 2D array; np.array([[x1, x2]])
        if predlog(np.array([[x1, x2]]), alpha, ord)[0] == 0:
            plt.plot(x1, x2, marker='.', color="red", markersize=2)
        else:
            plt.plot(x1, x2, marker='.', color="black", markersize=2)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Logistic regression classifier")
plt.show()
