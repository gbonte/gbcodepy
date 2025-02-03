## "INFOF422 Statistical foundations of machine learning" course
## Python package gbcodepy
## Author: G. Bontempi

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.ensemble import RandomForestRegressor





np.random.seed(0)
tf.random.set_seed(0)
N = 1000
n = 4

X = np.random.randn(N, n)
Y = np.sin(2 * np.pi * X[:, 0]) + X[:, 0] + np.random.randn(N) * 0.1

Itr = np.arange(0, int(N/2))
Its = np.setdiff1d(np.arange(0, N), Itr)

Xtr = X[Itr, :]
Xts = X[Its, :]

Ytr = Y[Itr]
Yts = Y[Its]

num_epochs = 100

## regularizer_l2(w) means every coefficient in the weight matrix of the layer
# will  add  w*weight_value  to  the  total  loss  of  the  network.

model = keras.Sequential([
    layers.Dense(units=10, activation="relu",
                 kernel_regularizer=regularizers.l2(0.01),
                 input_shape=(Xtr.shape[1],)),
    layers.Dropout(rate=0.5),
    layers.Dense(units=20, activation="relu",
                 kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)),
    layers.Dropout(rate=0.5),
    layers.Dense(units=1)
])

model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

history = model.fit(Xtr, Ytr,
                    validation_data=(Xts, Yts),
                    epochs=num_epochs, batch_size=1, verbose=1)

Yhats = model.predict(Xts)
NMSE = np.mean((Yts - Yhats.flatten())**2) / np.var(Yts)

Yhatr = model.predict(Xtr)
print(np.mean((Ytr - Yhatr.flatten())**2) / np.var(Ytr))

rf = RandomForestRegressor()
rf.fit(Xtr, Ytr)
Yhats2 = rf.predict(Xts)
NMSE2 = np.mean((Yts - Yhats2)**2) / np.var(Yts)

print("NMSE testset: DNN= ", NMSE, " RF= ", NMSE2, "\n")
