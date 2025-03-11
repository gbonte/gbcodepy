# "Statistical foundations of machine learning" software
# package gbcodepy 
# Author: G. Bontempi

#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


N = 10000

X1 = np.random.randn(N)
X2 = np.random.randn(N)

Y = X1 + X2

I = np.where(np.abs(Y) < 0.01)[0]

plt.figure()
plt.scatter(X1, X2,color="black")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("COLLIDER (y=x1+x2): conditioned on y=0 ")
plt.scatter(X1[I], X2[I], color="red")

D = pd.DataFrame({'X1': X1, 'X2': X2})
# Column names are already set by the keys in the DataFrame creation
reg1 = smf.ols(formula="X2 ~ X1", data=D).fit()

# Plot the regression line: get current x-axis limits for line extension
x_vals = np.array(plt.gca().get_xlim())
y_vals = reg1.params['Intercept'] + reg1.params['X1'] * x_vals
plt.plot(x_vals, y_vals, color="black", linewidth=2)

D2 = pd.DataFrame({'X1': X1[I], 'X2': X2[I]})
reg2 = smf.ols(formula="X2 ~ X1", data=D2).fit()

# Plot the regression line for the conditioned data using current x-axis limits
x_vals2 = np.array(plt.gca().get_xlim())
y_vals2 = reg2.params['Intercept'] + reg2.params['X1'] * x_vals2
plt.plot(x_vals2, y_vals2, color="red", linewidth=2)

plt.show()
