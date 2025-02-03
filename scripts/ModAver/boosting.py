# "INFOF422 Statistical foundations of machine learning" course
# Python package gbcodepy
# Author: G. Bontempi
#
# script boosting.py

# Import required libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# -------------------------------------------------
# Create synthetic dataset equivalent to Pima.tr
# Note: In R the dataset Pima.tr is loaded from the mlbench package.
# Here we generate a synthetic dataset with the same structure and columns.
np.random.seed(0)  # Seed for dataset generation reproducibility
N_total = 2000  # total number of observations (should be >= 40)
Pima_tr = pd.DataFrame({
    'npreg': np.random.randint(0, 10, N_total),
    'glu':   np.random.randint(80, 200, N_total),
    'bp':    np.random.randint(60, 130, N_total),
    'skin':  np.random.randint(20, 50, N_total),
    'bmi':   np.random.uniform(18, 40, N_total),
    'ped':   np.random.uniform(0, 3, N_total),
    'age':   np.random.randint(20, 80, N_total),
    'type':  np.random.choice(['Yes', 'No'], N_total)
})

# -------------------------------------------------
# Preprocess the data as in R: convert type to numeric: 1 if "Yes", -1 otherwise
Pima_tr['type'] = 2 * (Pima_tr['type'] == 'Yes').astype(int) - 1

# -------------------------------------------------
# Define training and test sets
N_tr = 40
N = len(Pima_tr)
train = Pima_tr.iloc[0:N_tr].copy()
ww = np.repeat(1/N_tr, N_tr)
ww=ww/sum(ww)

test = Pima_tr.iloc[N_tr:N].copy()

# -------------------------------------------------
# Number of boosting iterations
m = 25

misc = np.full(m, np.nan)
alpha = np.full(m, np.nan)
# For consistency with R's set.seed(555) in the first tree, we set the seed here:
np.random.seed(555)
# Build the first tree (single tree) model using sample weights adjusted by N_tr (w*N_tr)
# Use DecisionTreeClassifier with min_samples_split equivalent to mincut=10 in R's tree.control
features = ['npreg', 'glu', 'bp', 'skin', 'bmi', 'ped', 'age']
X_train = train[features].values
y_train = train['type'].values
X_test  = test[features].values
y_test  = test['type'].values

tree_model = DecisionTreeClassifier(min_samples_split=10, random_state=555)
# Multiply weights by N_tr as in R: weights=w*N.tr
tree_model.fit(X_train, y_train, sample_weight=ww * N_tr)

# Predict on test set and take the sign of predictions
pred = np.sign(tree_model.predict(X_test))
# Compute misclassification error for the single tree model
misc_tree = np.sum((y_test != np.sign(pred)).astype(int)) / len(pred)

# -------------------------------------------------
# Boosting algorithm initialization for test predictions
pred_test = np.zeros(len(X_test))

# Boosting iterations
for j in range(m):
    # Reset seed for reproducibility in each iteration (as in R: set.seed(555))
    np.random.seed(555)
    # Sample indices with replacement from training data according to weights w
    I = np.random.choice(np.arange(N_tr), size=N_tr, replace=True, p=ww/sum(ww))
    # Build bootstrap sample
    train_boot = train.iloc[I].copy()
    X_boot = train_boot[features].values
    y_boot = train_boot['type'].values

    # Fit tree model on bootstrap sample without weights (control parameters preserved)
    tree_model = DecisionTreeClassifier(min_samples_split=5, random_state=555)
    tree_model.fit(X_boot, y_boot)
    
    # Predict on the entire training set and take sign to get predictions
    pred_train = np.sign(tree_model.predict(X_train))
    
    # Compute weighted misclassification error on training data
    error_ind = (y_train != pred_train).astype(int)
    misc[j] = np.sum(ww * error_ind) / np.sum(ww)
    
    # Compute alpha for the boosting iteration
    alpha[j] = np.log((1 - misc[j]) / misc[j])
    
    # Update weights: multiply by exp(alpha * error indicator)
    ww = ww * np.exp(alpha[j] * error_ind)
    
    # Update the aggregated prediction on the test set
    pred_test = pred_test + alpha[j] * tree_model.predict(X_test)
    
    # If misclassification error is too high, reset weights to uniform distribution
    if misc[j] >= 0.49:
        w = np.repeat(1/N_tr, N_tr)

# -------------------------------------------------
# Compute misclassification error for the boosting model on the test set
misc_boosting = np.sum((y_test != np.sign(pred_test)).astype(int)) / len(pred_test)

# Print the misclassification errors
print("Misclassification single tree=", misc_tree)
print("Misclassification boosting=", misc_boosting)
