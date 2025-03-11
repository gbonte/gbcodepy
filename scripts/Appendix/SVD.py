import numpy as np
import matplotlib.pyplot as plt

# rm(list=ls())
# In Python, explicit clearing of the namespace is not required.

# set.seed(1)
np.random.seed(1)

# par(mfrow=c(1,2))
# In Python, we will create a figure with 1 row and 2 columns for subplots.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

N = 40  # number of examples
n = 20  # number of features

# X=array(rnorm(N*n),c(N,n))
X = np.random.randn(N, n)  # original data set

# S=svd(X)
# u: matrix whose columns contain the left singular vectors of x, present if nu > 0. Dimension (N, min(N, n)).
# d: vector containing the singular values of x, of length min(N, n), sorted decreasingly.
# v: matrix whose columns contain the right singular vectors of x, present if nv > 0. Dimension (n, min(N, n)).
U, s, Vh = np.linalg.svd(X, full_matrices=False)
V = Vh.T

# XX=X*0
XX = X * 0

# 
# ## X= \sum_{j=1}^min(N,n) \sigma_j U_j V_j^T 
# ## where U_j is the jth column of U and V_j is the jth column of V (or the jth row of V^T) 
for j in range(min(N, n)):
    XX = XX + s[j] * np.outer(U[:, j], V[:, j])

# cat("Error reconstructed with", min(N,n), "components",
#     mean((X-XX)^2),"\n")
error_full = np.mean((X - XX) ** 2)
print("Error reconstructed with", min(N, n), "components", error_full)

RecErr = []
NormErr = []

for k in range(1, min(N, n) + 1):
    # ## approximation with k components
    print("\n--------")
    XX = X * 0
    
    for j in range(k):
        XX = XX + s[j] * np.outer(U[:, j], V[:, j])
    
    # ## check of relation above
    frob_norm_error = np.linalg.norm(X - XX, 'fro')
    print("Error reconstructed with", k, "components", frob_norm_error)
    if k > 1:
        Z = np.dot(U[:, :k], np.diag(s[:k]))
        # ## [N,k] latent representation
        
        Xr = np.dot(Z, V[:, :k].T)
        # ## reconstructed matrix [N, m]
        
        print("Error reconstructed with", k, "components", np.linalg.norm(X - XX, 'fro'))
    RecErr.append(np.linalg.norm(X - XX, 'fro'))
    NormErr.append(np.linalg.norm(X - XX, 2))

# plot(1:min(N,n),RecErr,type="l",
#      main="Rank-k approximation",
#      xlab="# components", ylab="Frobenius norm of error")
axes[0].plot(range(1, min(N, n) + 1), RecErr, linestyle='-', marker='') 
axes[0].set_title("Rank-k approximation")
axes[0].set_xlabel("# components")
axes[0].set_ylabel("Frobenius norm of error")

# plot(1:min(N,n),NormErr,type="l",
#      main="Rank-k approximation",
#      xlab="# components", ylab="Spectral norm of error")
axes[1].plot(range(1, min(N, n) + 1), NormErr, linestyle='-', marker='') 
axes[1].set_title("Rank-k approximation")
axes[1].set_xlabel("# components")
axes[1].set_ylabel("Spectral norm of error")

# points(c(S$d[2:length(S$d)]),col="red")
# In R, points() with a single vector argument plots the points with the x-coordinate given by the index.
# We replicate this by plotting red points with x-coordinates starting at 1.
red_points_x = list(range(1, len(s[1:]) + 1))
red_points_y = s[1:]
axes[1].plot(red_points_x, red_points_y, 'ro')

plt.tight_layout()
plt.show()
