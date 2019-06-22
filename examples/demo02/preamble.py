# This script generates input files used to run MCMC from the shell prompt.

import numpy as np

import MCcubed as mc3


# The modeling function:
def quad(p, x):
    """
    Quadratic polynomial function.

    Parameters
        p: Polynomial constant, linear, and quadratic coefficients.
        x: Array of dependent variables where to evaluate the polynomial.
    Returns
        y: Polinomial evaluated at x:  y = p0 + p1*x + p2*x^2
    """
    y = p[0] + p[1]*x + p[2]*x**2.0
    return y


# Create a synthetic dataset using a quadratic polynomial curve:
x  = np.linspace(0, 10, 1000)         # Independent model variable
p0 = [3, -2.4, 0.5]                   # True-underlying model parameters
y  = quad(p0, x)                      # Noiseless model
uncert = np.sqrt(np.abs(y))           # Data points uncertainty
error  = np.random.normal(0, uncert)  # Noise for the data
data   = y + error                    # Noisy data set

# data.npz contains the data and uncertainty arrays:
mc3.utils.savebin([data, uncert], 'data.npz')
# indp.npz contains the list of additional arguments for the model:
mc3.utils.savebin([x],            'indp.npz')
