import numpy as np


N = 50000
Sigma = np.array([[1, 0.5, 0.3],
                  [0.5, 1, 0.3],
                  [0.3, 0.3, 1]])  # %Correlation matrix
A = np.linalg.cholesky(Sigma).T  # %Cholesky decomposition

D = np.random.randn(N, 3)  # %Random data in three columns each for X,Y and Z
Dc = D @ A  # %Correlated matrix Rc=[X Y Z]

# var(U*z)= U^T var(z) U =U^T U = C
print("\n ---\n Target Sigma=\n")
print(Sigma)
print("\n ---\n Estimated Sigma=\n")
print(np.cov(Dc, rowvar=False))
