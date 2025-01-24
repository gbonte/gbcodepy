# "INFOF422 Statistical foundations of machine learning" course
# lin_rls.py
# Original Author: G. Bontempi
# Translated to Python

import numpy as np
import matplotlib.pyplot as plt

def rls(x, y, t, P, mu=1):
    x = np.array(x).reshape(-1, 1)
    P_new = (P - (P @ x @ x.T @ P) / float(1 + x.T @ P @ x)) / mu
    ga = P_new @ x
    epsi = y - float(x.T @ t)
    t_new = t + ga * epsi
    return t_new, P_new

# Generate data
X = np.arange(-np.pi, np.pi, 0.02)
N = len(X)
y = np.sin(X) + 0.1 * np.random.normal(size=N)

# Initialize parameters
n = 1
t = np.zeros((2, 1))
P = 500 * np.eye(n + 1)
mu = 0.97

#plt.ion()  # Turn on interactive mode

for i in range(N):
    # Create input vector [1, X[i]]
    x_input = np.array([[1], [X[i]]])
    
    # RLS update
    t, P = rls(x_input, y[i], t, P, mu)
    
    if i % 10 == 0:
        fig, ax = plt.subplots(figsize=(8, 8))  # Create figure and axes
        #plt.clf()  # Clear the current figure
        ax.plot(X[:i+1], y[:i+1], 'b.')
        plt.xlim(-4, 4)
        plt.ylim(-2, 2)
        plt.title(f"RLS Forgetting factor mu: {mu}")
        plt.xlabel("x")
        plt.ylabel("y")
        
        # Generate prediction line
        X_plot = X[:i+1]
        X_augmented = np.column_stack((np.ones_like(X_plot), X_plot))
        y_pred = X_augmented @ t
        ax.plot(X_plot, y_pred, 'r-')
        
        display(fig)
        clear_output(wait=True)  # Clear the output
    
        plt.close(fig)  # Close the figure
        time.sleep(0.5)
        #input(" ")

#plt.tight_layout()
#plt.show()
