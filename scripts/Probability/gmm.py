import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def dmixture(x1, x2, w, centers, sds):
    """
    Mixture of Gaussians density.
    """
    m = len(w)
    dens = 0
    for i in range(m):
        mean = centers[i]
        cov = np.diag(sds[i])
        dens += w[i] * multivariate_normal.pdf([x1, x2], mean=mean, cov=cov)
    return dens

def GMM(N, n, w, centers, sds):
    """
    Random sampling from mixture of Gaussians.
    """
    m = len(w)  # number of mixture components
    X = []
    W = []  # keep trace of which component was sampled
    for _ in range(N):
        whichm = np.random.choice(m, p=w)
        W.append(whichm)
        if n > 1:
            sample = np.random.multivariate_normal(mean=centers[whichm], cov=np.diag(sds[whichm]))
            X.append(sample)
        else:
            sample = np.random.normal(loc=centers[whichm], scale=sds[whichm])
            X.append(sample)
    X = np.array(X)
    return {'X': X, 'W': W, 'centers': centers, 'sds': sds}

# Set random seed for reproducibility
np.random.seed(0)

# Define colors
cols = ["red", "green", "blue", "magenta", "black", "yellow"] * 3

# Parameters
N = 2000
m = 3
n = 2

# Generate weights
w = np.random.uniform(size=m)
w /= np.sum(w)

# Generate centers and standard deviations
centers = np.random.normal(scale=1.5, size=(m, n))
sds = np.random.uniform(0, 0.6, size=(m, n))

# Define grid for density plot
x1 = np.arange(-3, 3.1, 0.1)
x2 = np.arange(-3, 3.1, 0.1)
X1, X2 = np.meshgrid(x1, x2)
D = np.array([dmixture(x, y, w, centers, sds) for x, y in zip(np.ravel(X1), np.ravel(X2))])
D = D.reshape(X1.shape)

# Sample data from GMM
D_sample = GMM(N, n, w, centers, sds)

# Plotting
fig = plt.figure(figsize=(12, 6))

# First subplot: 3D density
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X1, X2, D, alpha=0.5)
ax1.set_title(f"Mixture of Gaussians with {m} components")
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('Density')

# Second subplot: Sampled points
ax2 = fig.add_subplot(1, 2, 2)
scatter = ax2.scatter(D_sample['X'][:, 0], D_sample['X'][:, 1], c=[cols[w] for w in D_sample['W']], s=10)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Samples')

plt.tight_layout()
plt.show()
