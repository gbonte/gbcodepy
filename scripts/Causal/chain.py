import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


N = 100000

X1 = np.random.normal(loc=0, scale=0.5, size=N)

Y = X1 + np.random.normal(loc=0, scale=0.2, size=N)
X2 = 2 * Y + np.random.normal(loc=0, scale=0.2, size=N)

I = np.where(np.abs(Y) < 0.001)[0]

plt.figure()
plt.scatter(X1, X2,color="black")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("CHAIN (y=x1+e1, x2=2y+e2): conditioned on y=0")
plt.scatter(X1[I], X2[I], color="red")

D = pd.DataFrame({'X1': X1, 'X2': X2})
reg1 = smf.ols(formula="X2 ~ X1", data=D).fit()

x_vals = np.array([X1.min(), X1.max()])
y_vals = reg1.params['Intercept'] + reg1.params['X1'] * x_vals
plt.plot(x_vals, y_vals, color="black", linewidth=2)

D2 = pd.DataFrame({'X1': X1[I], 'X2': X2[I]})
reg2 = smf.ols(formula="X2 ~ X1", data=D2).fit()

y_vals2 = reg2.params['Intercept'] + reg2.params['X1'] * x_vals
plt.plot(x_vals, y_vals2, color="red", linewidth=2)

plt.show()

