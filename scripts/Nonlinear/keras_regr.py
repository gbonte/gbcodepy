import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.ensemble import RandomForestRegressor

# "INFOF422 Statistical foundations of machine learning" course
# R package gbcode 
# Author: G. Bontempi

# rm(list=ls())
# In Python, we do not typically clear the entire workspace; no equivalent operation is performed here

np.random.seed(0)
N = 1000
n = 4

X = np.random.randn(N * n).reshape(N, n)
Y = np.sin(2 * np.pi * np.mean(np.column_stack((X[:, 0], X[:, 0])), axis=1)) + X[:, 0] + np.random.randn(N) * 0.1

Itr = np.arange(0, N//2)  # Equivalent to R's 1:(N/2), adjusted for 0-indexing in Python
Its = np.setdiff1d(np.arange(0, N), Itr)

Xtr = X[Itr, :]
Xts = X[Its, :]

Ytr = Y[Itr]
Yts = Y[Its]

num_epochs = 100

# regularizer_l2(w) means every coefficient in the weight matrix of the layer
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

model.compile(optimizer="rmsprop",
              loss="mse",
              metrics=["mae"])

history = model.fit(Xtr, Ytr,
                    validation_data=(Xts, Yts),
                    epochs=num_epochs, batch_size=1, verbose=1)

Yhats = model.predict(Xts).flatten()
NMSE = np.mean((Yts - Yhats)**2) / np.var(Yts)

Yhatr = model.predict(Xtr).flatten()
print(np.mean((Ytr - Yhatr)**2) / np.var(Ytr))

def predRF(X_train, Y_train, X_test):
    rf = RandomForestRegressor()
    rf.fit(X_train, Y_train)
    return rf.predict(X_test)
    

Yhats2 = predRF( Xtr, Ytr, Xts)
NMSE2 = np.mean((Yts - Yhats2)**2) / np.var(Yts)

print("NMSE testset: DNN= ", NMSE, " RF= ", NMSE2)
