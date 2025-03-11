# ## "INFOF422 Statistical foundations of machine learning" course
# ## package gbcodepy 
# ## Author: G. Bontempi 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# library(tree)  # In R: Load tree package. In Python, we use DecisionTreeRegressor from sklearn.tree

N = 2000
def dopler(x):
    return 20 * np.sqrt(x * (1 - x)) * np.sin(2 * np.pi * 1.05 / (x + 0.05))

np.random.seed(0)
x = np.sort(np.random.uniform(low=0.12, high=1, size=N))
sigma = 1  # Defining sigma since it is not specified in the original code
y = dopler(x) + np.random.normal(loc=0, scale=sigma, size=N)
x_ts = np.sort(np.random.uniform(low=0.12, high=1, size=N))
y_ts = dopler(x_ts)
plt.figure()
plt.plot(x, y, linestyle='-', marker='')
plt.show()

d = pd.DataFrame({'y': y, 'x': x})
d.columns = ["Y","X"]

# Create D data structure for test data plotting as used later: D$x.ts and D$y.ts in R code.
D = pd.DataFrame({'x.ts': x_ts, 'y.ts': y_ts})

for number_nodes in range(1, 31):
    X_train = d[["X"]]
    y_train = d["Y"]
    mod_tree = DecisionTreeRegressor(min_samples_leaf=number_nodes, min_impurity_decrease=0)
    mod_tree.fit(X_train, y_train)
    
    d_ts = pd.DataFrame({'y.ts': y_ts, 'x.ts': x_ts})
    # names(d.ts) <- c("Y", "X")
    d_ts.columns = ["Y", "X"]
    
    p = mod_tree.predict(d_ts[["X"]])
    
    plt.figure()
    plt.plot(D["x.ts"], D["y.ts"], linestyle='-', marker='', label="True")
    plt.title("Min number points per leaf=" + str(number_nodes))
    plt.plot(D["x.ts"], p, color="red", label="Predicted")
    plt.legend()
    plt.show()
    
