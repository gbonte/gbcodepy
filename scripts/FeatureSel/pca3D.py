import numpy as np
import plotly.graph_objects as go

# "INFOF422 Statistical foundations of machine learning" course
# R package gbcode
# Author: G. Bontempi

# Clearing workspace (not required in Python)

# Importing plotly (already imported above)

N = 500

# Generate random data analogous to rnorm(N, sd=1)
X1 = np.random.normal(loc=0, scale=1, size=N)
X2 = np.random.normal(loc=0, scale=1, size=N)
# Generate X3 equivalent to: rnorm(1)*X1 + rnorm(1)*X2 + rnorm(N, sd=5.2)
a_random = np.random.normal(loc=0, scale=1, size=1)[0]
b_random = np.random.normal(loc=0, scale=1, size=1)[0]
noise = np.random.normal(loc=0, scale=5.2, size=N)
X3 = a_random * X1 + b_random * X2 + noise

# Create matrix by column-binding X1, X2, X3 and scale each column (center and scale by sample standard deviation)
X = np.column_stack((X1, X2, X3))
Xtilde = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)

x1 = np.linspace(-3, 3, 30)
x2 = x1.copy()

# Singular Value Decomposition of Xtilde
U, s, Vt = np.linalg.svd(Xtilde, full_matrices=False)
S = {'u': U, 'd': s, 'v': Vt.T}

# Eigen decomposition of (Xtilde.T @ Xtilde)
E_values, E_vectors = np.linalg.eig(np.dot(Xtilde.T, Xtilde))
# Check that (Xtilde.T @ Xtilde) equals E_vectors @ diag(E_values) @ E_vectors.T
# and that the squared singular values S['d']**2 equal E_values

# V2 is the first two columns of S$v
V2 = S['v'][:, :2]

# EV is set to be the first two columns of S$v (overriding the eigen() result)
EV = S['v'][:, :2]

# VY is a (2x1) array composed of the third row from EV (note: Python uses 0-indexing)
VY = np.array([[EV[2, 0]], [EV[2, 1]]])
# VX is the transpose of the top-left 2x2 submatrix of EV
VX = EV[0:2, 0:2].T

# Solve for b: b = inv(VX) @ VY
b = np.linalg.solve(VX, VY)

# Compute the projection Z of Xtilde onto V2
Z = np.dot(Xtilde, V2)

# Reconstruct Xtilde2 using the projection Z and V2
Xtilde2 = np.dot(Z, V2.T)
RecE = Xtilde - Xtilde2  # reconstruction error

# Print reconstruction error: mean of the sum of squares of each row vs S$d[3]^2/N (Python index adjustment: s[2])
print("Reconstruction error=", np.mean(np.sum(RecE**2, axis=1)), ":", S['d'][2]**2 / N)

# Define the function f equivalent to R's f <- function(x1, x2, a, b) { a*x1 + b*x2 }
def f(x1, x2, a, b):
    return a * x1 + b * x2

# In the outer product, R calls f(x1, x2, b[2], b[1]). Note: in R b[1] is the first element 
# and b[2] is the second element; in Python, after flattening, b[0] corresponds to R's b[1] and b[1] to R's b[2].
b_flat = b.flatten()
a_param = b_flat[1]  # corresponds to b[2] in R
b_param = b_flat[0]  # corresponds to b[1] in R

# Compute z using broadcasting to mimic R's outer function: f(x1, x2, a, b)
# Resulting z[i,j] = a_param*x1[i] + b_param*x2[j]
z = a_param * x1[:, np.newaxis] + b_param * x2[np.newaxis, :]
z[np.isnan(z)] = 1

# Set graphical parameters analogous to op <- par(bg = "white")
op = {"bg": "white"}

# Create a plotly figure and add a surface plot for z using x1 and x2 grids
fig = go.Figure()
fig.add_trace(go.Surface(x=x1, y=x2, z=z))

# Add markers for the original data points from Xtilde
fig.add_trace(go.Scatter3d(x=Xtilde[:, 0], y=Xtilde[:, 1], z=Xtilde[:, 2],
                           mode='markers',
                           marker=dict(color='red', size=120)))
# Add markers for the reconstructed data points from Xtilde2
fig.add_trace(go.Scatter3d(x=Xtilde2[:, 0], y=Xtilde2[:, 1], z=Xtilde2[:, 2],
                           mode='markers',
                           marker=dict(color='black', size=120)))

# The following segment is commented out in the R code and is preserved as a comment in Python
# fig.add_trace(go.Scatter3d(x=[Xtilde[0,0], Dc[0,0]],
#                            y=[Xtilde[0,1], Dc[0,2]],
#                            z=[Xtilde[0,2], Dc[0,3]],
#                            mode='lines',
#                            marker=dict(color='green')))

fig.show()
