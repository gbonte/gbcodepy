import numpy as np
import scipy.integrate as integrate

def normal_pdf(x, mu, sigma):
  """Probability density function of a Normal distribution."""
  return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def uniform_pdf(x,a,b):
  if a <= x <= b:
    return 1 / (b - a)
  else:
    return 0

def compute_mean_by_integration(pdf,mu, sigma, a, b):
  """Computes the mean of a Normal distribution using numerical integration.

  Args:
    mu: Mean of the Normal distribution.
    sigma: Standard deviation of the Normal distribution.
    a: Lower limit of integration.
    b: Upper limit of integration.

  Returns:
    The computed mean.
  """
  if pdf==normal_pdf:
    result = integrate.quad(lambda x: x * pdf(x, mu, sigma), a, b)
  if pdf==uniform_pdf:
    result = integrate.quad(lambda x: x * pdf(x, a, b), a, b)
  return result[0]  # Extract the computed value from the result tupl


def compute_variance_by_integration(pdf, mu, sigma, a, b):
    """Computes the variance of a Normal distribution using numerical integration.
    Args:
        mu: Mean of the Normal distribution.
        sigma: Standard deviation of the Normal distribution.
        a: Lower limit of integration.
        b: Upper limit of integration.
    Returns:
        The computed variance.
    """
    # Calculate E[X^2]
    expected_x_squared = integrate.quad(lambda x: x**2 * pdf(x, mu, sigma), a, b)[0]
    # Calculate (E[X])^2
    expected_x_squared_2 = integrate.quad(lambda x: x * pdf(x, mu, sigma), a, b)[0] ** 2
    
    # Variance = E[X^2] - (E[X])^2
    variance = expected_x_squared - expected_x_squared_2
    
    return variance
  
def compute_variance_monte_carlo(mu, sigma, num_samples):
    """Computes the variance of a Normal distribution using Monte Carlo simulation.

    Args:
        mu: Mean of the Normal distribution.
        sigma: Standard deviation of the Normal distribution.
        num_samples: Number of samples to generate.

    Returns:
        The estimated variance.
    """

    # Generate random samples from the Normal distribution
    samples = np.random.normal(mu, sigma, num_samples)


    # Calculate the average of the samples
    avg = np.mean(samples)
    # Calculate the variance of the samples
    variance = np.var(samples)

    return avg, variance
  
# Example usage
mu = 0.35  # Mean of the distribution
sigma = 0.4  # Standard deviation of the distribution
unif_l=-1
unif_u=1

a = -5  # Lower limit of integration
b = 5   # Upper limit of integration



num_samples=1000
avg_mc, var_mc =compute_variance_monte_carlo(mu, sigma, num_samples)

computed_mean = compute_mean_by_integration(normal_pdf,mu, sigma, a, b)
print(f"Mean: {mu:.2f}, Mean computed with integration: {computed_mean:.2f}, Mean computed with MC: {avg_mc:.2f}")


computed_variance = compute_variance_by_integration(normal_pdf,mu, sigma, a, b)

print(f"""Variance: {pow(sigma,2):.2f}, Variance computed with integration: {computed_variance:.2f},
    Variance computed with MC: {var_mc:.2f}""")


