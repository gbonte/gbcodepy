## # "Statistical foundations of machine learning" software
# ttorch.py
# Author: G. Bontempi
# assessment of some torch FNN in a regression task

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


import sys
import os

# Get the absolute path of the directory containing the script
#script_dir = os.path.abspath('~/Dropbox/bontempi_office/Rlang/gbcode2/inst/py')  # Replace with actual path


# Add the directory to the Python path
#sys.path.append(script_dir)


from importlib import reload

from pred import predpy
reload(pred)

num_samples = 1000
num_features = 15

# Generate random input features (X)
X = np.random.rand(num_samples, num_features)

# Generate random target variable (y) with a linear relationship and some noise
# Adjust coefficients and noise level as needed
y = 2 * pow(X[:, 0],2) + 3 * np.log(abs(X[:, 1])) - 1 * X[:, 2] + np.random.normal(0, 0.5, num_samples)  

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


yhat=predpy("torch_regr",X_train,y_train,X_test,params={ "nepochs":1500})
print(f'Torch MSE: {np.mean(pow(y_test-yhat,2)):.4f}')

yhat=predpy("keras0_regr",X_train,y_train,X_test)
print(f'Keras MSE: {np.mean(pow(y_test-yhat,2)):.4f}')


yhat=predpy("rf_regr",X_train,y_train,X_test)
print(f'RF MSE: {np.mean(pow(y_test-yhat,2)):.4f}')

