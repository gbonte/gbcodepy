
# ## "INFOF422 Statistical foundations of machine learning" course
# Python package gbcodepy 
# ## Author: G. Bontempi
#
# ## script arcing.py

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset and construct a DataFrame with the appropriate columns.
# We select the first 9 features and rename them to match the R code:
# "Cl.thickness", "Cell.size", "Cell.shape", "Marg.adhesion", "Epith.c.size", "Bare.nuclei",
# "Bl.cromatin", "Normal.nucleoli", "Mitoses".
data = load_breast_cancer()
# Create a DataFrame using the first 9 features
features = ["Cl.thickness", "Cell.size", "Cell.shape", "Marg.adhesion", "Epith.c.size", 
            "Bare.nuclei", "Bl.cromatin", "Normal.nucleoli", "Mitoses"]
df_features = pd.DataFrame(data.data[:, :9], columns=features)

# Create the 'Class' column based on the target.
# In the sklearn dataset, target = 1 corresponds to 'benign' and target = 0 to 'malignant'.
class_labels = np.where(data.target == 1, 'benign', 'malignant')
BreastCancer = df_features.copy()
BreastCancer['Class'] = class_labels


N_tr = 400

N = BreastCancer.shape[0]
# I<-BreastCancer$Class=='benign'
I = (BreastCancer['Class'] == 'benign')

BreastCancer.loc[I, 'Class2'] = 1
BreastCancer.loc[~I, 'Class2'] = -1


train = BreastCancer.iloc[0:N_tr].copy()

w = np.repeat(1/N_tr, N_tr)


test = BreastCancer.iloc[N_tr:].copy()


m = 25


misc = np.full(m, np.nan)
alpha = np.full(m, np.nan)


np.random.seed(555)

X_train = train[features].values
y_train = train['Class2'].values.astype(float)
tree_model = DecisionTreeRegressor(min_samples_split=5, random_state=555)
tree_model.fit(X_train, y_train, sample_weight=w * N_tr)

X_test = test[features].values
pred = np.sign(tree_model.predict(X_test))
misc_tree = np.sum(test['Class2'].values != pred) / len(pred)

pred_test = np.zeros(N - N_tr)
mi = np.zeros(N_tr)

# for (j in 1:m)
for j in range(m):
    # set.seed(555)
    np.random.seed(555)
    indices = np.arange(N_tr)
    I_boot = np.random.choice(indices, size=N_tr, replace=True, p=w)
    
    train_boot = train.iloc[I_boot].copy()
    X_train_boot = train_boot[features].values
    y_train_boot = train_boot['Class2'].values.astype(float)
    tree_model = DecisionTreeRegressor(min_samples_split=5, random_state=555)
    tree_model.fit(X_train_boot, y_train_boot)
    
    pred_train = np.sign(tree_model.predict(train[features].values))
    
    mi += (train['Class2'].values != pred_train).astype(int)
    
    w = (1 + mi**4) / np.sum(1 + mi**4)
    
    pred_test += np.sign(tree_model.predict(test[features].values))

pred_test = pred_test / m
misc_arcing = np.sum(test['Class2'].values != np.sign(pred_test)) / len(pred_test)

print("Misclassification single tree=", misc_tree)
print("Misclassification boosting=", misc_arcing)
