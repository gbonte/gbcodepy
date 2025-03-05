
import numpy as np
import matplotlib.pyplot as plt

## "INFOF422 Statistical foundations of machine learning" course
## package gbcodepy
## Author: G. Bontempi

## Visualization of leave-one-out strategy with a linear model

def f(x, ord):
    f = 1
    for i in range(1, ord + 1):
        f = f + (x ** i)
    return f

np.random.seed(0)

n = 1
N = 25

x = np.linspace(-2, 2, N)
N = len(x)
sd_w = 5

O = 3
Y = f(x, ord=O) + np.random.normal(0, sd_w, N)
data_tr = np.column_stack((Y, x))

X = np.column_stack((np.ones(N), x))

beta = np.linalg.solve(X.T @ X, X.T @ Y)
Yhat = X @ beta

for i in range(N):
    Xtri = np.delete(X, i, axis=0)
    Ytri = np.delete(Y, i)
    betai = np.linalg.solve(Xtri.T @ Xtri, Xtri.T @ Ytri)
    Yhati = X @ betai
    ei = (Y[i] - Yhat[i]) ** 2
    plt.scatter(x, Y, label='Observations',color='black')
    plt.scatter(x[i], Y[i], color="red", s=50, marker='o')
    plt.plot(x, Yhat, label='Fitted Line', linewidth=3)
    plt.plot(x, Yhati, color="red", linewidth=3,label='LOO fitted Line')
    plt.plot([x[i], x[i]], [Y[i], Yhati[i]], color="red", 
             linewidth=3, linestyle='dashed')
    plt.title(f"Squared loo residual={round(ei, 2)}")
    plt.legend()
    plt.show()
    input('')

