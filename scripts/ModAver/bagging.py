# ## "INFOF422 Statistical foundations of machine learning" course
# ## R package gbcode 
# ## Author: G. Bontempi
# ## script bagging.py

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import math
import warnings
from sklearn.exceptions import ConvergenceWarning

# Avoid warnings of convergence
warnings.filterwarnings("ignore", category=ConvergenceWarning)


N = 200
n = 3
np.random.seed(555)
X = np.random.normal(loc=0, scale=1, size=N * n)
X = np.reshape(X, (N, n))

Y = (X[:, 0] ** 2) + 5 * X[:, 2] + 4 * np.log(np.abs(X[:, 1])) + np.random.normal(loc=0, scale=math.sqrt(0.25), size=N)

N_tr = int(N / 2)
I_ts = np.arange(N_tr, N) 

data_train = pd.DataFrame({
    'y': Y[:N_tr],
    'x1': X[:N_tr, 0],
    'x2': X[:N_tr, 1],
    'x3': X[:N_tr, 2]
})

data_test = pd.DataFrame({
    'y': Y[I_ts],
    'x1': X[I_ts, 0],
    'x2': X[I_ts, 1],
    'x3': X[I_ts, 2]
})


################################## Single NNET
np.random.seed(555)
# In MLPRegressor, hidden_layer_sizes=(25,) specifies one hidden layer with 25 neurons.
# activation='logistic' is selected to mimic the logistic activation of nnet's hidden units.
model_nn = MLPRegressor(hidden_layer_sizes=(25,), max_iter=10000, activation='logistic', solver='adam',
                        random_state=555, verbose=False)
model_nn.fit(data_train[['x1', 'x2', 'x3']], data_train['y'])
predict_nn_1 = model_nn.predict(data_test[['x1', 'x2', 'x3']])
MSE_nn1 = np.mean((data_test['y'] - predict_nn_1) ** 2)
print("Test error single NNET=" + str(MSE_nn1))


#################################
B = 50
predict_nn = np.zeros((len(I_ts), B))
predict_bag = np.zeros(len(I_ts))
MSE_nn = np.zeros(B)
for b in range(1, B + 1):
    np.random.seed(b)
    I_D_b = np.random.choice(np.arange(N_tr), size=N_tr, replace=True)
    data_bagging = pd.DataFrame({
        'y': Y[I_D_b],
        'x1': X[I_D_b, 0],
        'x2': X[I_D_b, 1],
        'x3': X[I_D_b, 2]
    })
    np.random.seed(555)
    model_nn = MLPRegressor(hidden_layer_sizes=(25,), max_iter=1000, 
                            activation='logistic', solver='adam', learning_rate_init=0.01,
                            random_state=555, verbose=False)
    model_nn.fit(data_bagging[['x1', 'x2', 'x3']], data_bagging['y'])
    predict_nn[:, b - 1] = model_nn.predict(data_test[['x1', 'x2', 'x3']])
    MSE_nn[b - 1] = np.mean((data_test['y'] - predict_nn[:, b - 1]) ** 2)

for i in range(len(I_ts)):
    predict_bag[i] = np.mean(predict_nn[i, :])

MSE_bag = np.mean((data_test['y'] - predict_bag) ** 2)
print("Test error bagging NNET=" + str(MSE_bag))

print("Average test error bagging NNET=" + str(np.mean(MSE_nn)))

plt.hist(MSE_nn, alpha=0.75)
plt.title('')
plt.axvline(x=MSE_bag, linewidth=3, color='r')
plt.axvline(x=np.mean(MSE_nn), linewidth=3, color='g', linestyle='dashed')
plt.show()
