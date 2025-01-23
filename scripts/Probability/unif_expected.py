import numpy as np
import scipy.integrate as integrate


def uniform_pdf(x,a,b):
  if a <= x <= b:
    return 1 / (b - a)
  else:
    return 0

def compute_mean_by_integration(lower, upper,a,b):
  """Computes the mean of a Normal distribution using numerical integration.

  Args:
    mu: Mean of the Normal distribution.
    sigma: Standard deviation of the Normal distribution.
    a: Lower limit of integration.
    b: Upper limit of integration.

  Returns:
    The computed mean.
  """

  
  result = integrate.quad(lambda x: x * uniform_pdf(x, lower, upper), a, b)
  return result[0]  # Extract the computed value from the result tupl


def compute_variance_by_integration(lower, upper,a,b):
    """Computes the variance of a uniform distribution using numerical integration.
    Args:
        
        a: Lower limit of integration.
        b: Upper limit of integration.
    Returns:
        The computed variance.
    """
    # Calculate E[X^2]
    expected_x_squared = integrate.quad(lambda x: x**2 * uniform_pdf(x, lower, upper), a, b)[0]
    # Calculate (E[X])^2
    expected_x_squared_2 = integrate.quad(lambda x: x * uniform_pdf(x, lower, upper), a, b)[0] ** 2
    
    # Variance = E[X^2] - (E[X])^2
    variance = expected_x_squared - expected_x_squared_2
    
    return variance
  
def compute_variance_monte_carlo(lower, upper, num_samples):
    """Computes the variance of a Uniform distribution using Monte Carlo simulation.

    Args:
        
        num_samples: Number of samples to generate.

    Returns:
        The estimated variance.
    """

    # Generate random samples from the Normal distribution
    samples = np.random.uniform(lower, upper, num_samples)

    # Calculate the average of the samples
    avg = np.mean(samples)
    # Calculate the variance of the samples
    variance = np.var(samples)

    return avg, variance
  
# Example usage

lower=-1
upper=3

a = -5  # Lower limit of integration
b = 5   # Upper limit of integration



num_samples=1000
avg_mc, var_mc =compute_variance_monte_carlo(lower,upper, num_samples)

computed_mean = compute_mean_by_integration(lower,upper, a, b)
print(f"Mean: {(lower+upper)/2:.2f}, Mean computed with integration: {computed_mean:.2f}, Mean computed with MC: {avg_mc:.2f}")


computed_variance = compute_variance_by_integration(lower,upper, a, b)

print(f"""Variance: {pow((upper-lower),2)/12:.2f}, Variance computed with integration: {computed_variance:.2f},
    Variance computed with MC: {var_mc:.2f}""")


