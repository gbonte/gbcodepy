import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb



# Mixture of gaussians density
def dmixture(x1, x2, w, centers, sds):
    print(w)
    m = w.shape[0]
    dens = 0
    
    for i in range(m):
        dens += w[i] * multivariate_normal.pdf([x1, x2], mean=centers[i], cov=np.diag(sds[i]))
    
    return dens

# Random sampling from mixture of gaussians
def GMM(N, n, w, centers, sds):
    m = w.shape[0]
    X = []
    W = []  # Keep track of which component was sampled
    
    for _ in range(N):
        whichm = np.random.choice(m, p=w)
        W.append(whichm)
        if n > 1:
            X.append(multivariate_normal.rvs(mean=centers[whichm], cov=np.diag(sds[whichm])))
        else:
            X.append(np.random.normal(loc=centers[whichm], scale=sds[whichm]))
    
    return {'X': np.array(X), 'W': np.array(W), 'centers': centers, 'sds': sds}

np.random.seed(0)
cols = ['red', 'green', 'blue', 'magenta', 'black', 'yellow'] * 3
N = 2000
m = 3
n = 2
w = np.random.uniform(size=m)
w = w / np.sum(w)
centers = np.random.normal(scale=1.5, size=(m, n))
sds = np.random.uniform(0, 0.6, size=(m, n))

x1 = np.arange(-3, 3, 0.1)
x2 = np.arange(-3, 3, 0.1)
X1, X2 = np.meshgrid(x1, x2)



D = GMM(N, 2, w, centers, sds)

vd = np.vectorize(dmixture)
print(w)
d = vd(X1, X2, w, centers, sds)

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X1, X2, d, cmap='viridis')
ax1.set_title(f"Mixture of gaussians with {m} components")

ax2 = fig.add_subplot(122)
ax2.scatter(D['X'][:, 0], D['X'][:, 1], c='gray', alpha=0.5)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Samples')

for class_ in range(m):
    w = np.where(D['W'] == class_)[0]
    ax2.scatter(D['X'][w, 0], D['X'][w, 1], c=cols[class_ + 2])

plt.tight_layout()
plt.show()

