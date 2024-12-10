import numpy as np
import matplotlib.pyplot as plt

# Bayesian parametric estimation of the mean of a normal variable with known variance

plt.ion()  # Turn on interactive mode

sigma = 1.5   # Known standard deviation
theta0 = 13   # A priori mean
sigma0 = 1    # A priori standard deviation
Nmax = 1000
DN = np.random.normal(0, sigma, Nmax)  # Data generation

# Computation of a posteriori distribution
for N in range(2, Nmax + 1, 50):
    # Plot of a posteriori theta estimation
    mu_hat = np.mean(DN[:N])
    sigma1 = np.sqrt(((sigma**2)/N) * ((sigma0**2)/(sigma0**2 + ((sigma**2)/N))))
    theta1 = (sigma1**2) * (theta0/(sigma0**2) + N*mu_hat/(sigma**2))
    
    I = np.linspace(-20, 20, 401)
    post = np.exp(-0.5 * ((I - theta1) / sigma1)**2) / (sigma1 * np.sqrt(2 * np.pi))
    
    plt.figure()
    plt.plot(I, post, color='blue', label='Posterior')
    plt.title(f"N={N}, thetahat(freq)={mu_hat:.3f}, thetahat(bay)={I[np.argmax(post)]:.3f}")
    plt.xlabel("theta")
    plt.ylabel("posterior distribution")
    
    # Plot of a priori theta estimation
    prior = np.exp(-0.5 * ((I - theta0) / sigma0)**2) / (sigma0 * np.sqrt(2 * np.pi))
    plt.plot(I, prior, color='red', label='Prior')
    
    # Plot of sampled data
    plt.scatter(DN[:N], np.zeros(N), color='black', s=1)
    
    plt.legend()
    plt.show()
    plt.pause(0.1)  # Pause to allow the plot to be displayed

plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the final plot open

