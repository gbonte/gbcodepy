import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import tree


sigma = 1

N = 1000

def dopler(x):
    return 20 * np.sqrt(x * (1 - x)) * np.sin(2 * np.pi * 1.05 / (x + 0.05))

np.random.seed(0)
x = np.sort(np.random.uniform(0.12, 1, N))
y = dopler(x) + np.random.normal(0, sigma, N)
x_ts = np.sort(np.random.uniform(0.12, 1, N))
y_ts = dopler(x_ts)
plt.figure()
plt.plot(x, y, linestyle='-', marker=None)
plt.show()

d = pd.DataFrame({'Y': y, 'X': x})
# names(d) <- c("Y","X") equivalent by constructing the DataFrame with the desired column names

MSE = []

Nn = np.arange(3, 51, 1)

# Since the original loop uses D$y.ts and D$x.ts, we create a dictionary D with these entries.
D = {'y.ts': y_ts, 'x.ts': x_ts}

def lazy_control(conIdPar=None, linIdPar=None, quaIdPar=None):
    return {'conIdPar': conIdPar, 'linIdPar': linIdPar, 'quaIdPar': quaIdPar}

class lazyModel:
    def __init__(self, formula, data, control):
        self.formula = formula
        self.data = data.copy()
        # Use the first element of linIdPar as the number of neighbors
        self.k = control['linIdPar'][0] if control['linIdPar'] is not None else 1
        
    def predict(self, newdata):
        predictions = []
        train_X = self.data['X'].values
        train_Y = self.data['Y'].values
        test_X = newdata['X'].values
        for x_val in test_X:
            distances = np.abs(train_X - x_val)
            knn_indices = np.argsort(distances)[:self.k]
            pred = np.mean(train_Y[knn_indices])
            predictions.append(pred)
        # Create an object p with attribute h that stores predictions.
        p = type('Prediction', (), {})()
        p.h = np.array(predictions)
        return p

def lazy(formula, data, control):
    return lazyModel(formula, data, control)

for number_neighbors in Nn:
    mod_lazy = lazy("Y~.", d,
                    control=lazy_control(conIdPar=None,
                                         linIdPar=(number_neighbors, number_neighbors),
                                         quaIdPar=None))
    
    d_ts = pd.DataFrame({'Y': D['y.ts'], 'X': D['x.ts']})
    p = mod_lazy.predict(d_ts)
    
    mse_value = np.mean((p.h - D['y.ts'])**2)
    MSE.append(mse_value)
    
    plt.figure()
    plt.plot(D['x.ts'], D['y.ts'], linestyle='-', marker=None, label="True")
    plt.plot(D['x.ts'], p.h, color="red", label="Predicted")
    plt.title("Number neighbors=" + str(number_neighbors))
    plt.legend()
    plt.show()
    time.sleep(0.1)

plt.figure()
plt.plot(Nn, MSE, linestyle='-')
plt.ylabel("MISE")
plt.xlabel("Number of neighbors")
plt.show()
