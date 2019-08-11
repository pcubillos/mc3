import numpy as np
import mc3


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

# For the sake of example, create a noisy synthetic dataset, in a real
# scenario you would get your dataset from your data analysis pipeline:
np.random.seed(3)
x = np.linspace(0, 10, 100)
p_true = [3.0, -2.4, 0.5]
y = quad(p_true, x)
uncert = np.sqrt(np.abs(y))
data = y + np.random.normal(0, uncert)

# Initial guess for fitting parameters:
params = np.array([10.0, -2.0, 0.1])
pstep  = np.array([0.03, 0.03, 0.05])

# Run the MCMC:
func = quad
mc3_results = mc3.sample(data, uncert, func, params, indparams=[x],
    pstep=pstep, sampler='snooker', nsamples=1e5, burnin=1000, ncpu=7)


# And now, some post processing:
import mc3.plots as mp
import mc3.utils as mu

# Output dict contains the entire sample (posterior), need to remove burn-in:
posterior, zchain, zmask = mu.burn(mc3_results)
bestp = mc3_results['bestp']
# Set parameter names:
pnames = ["constant", "linear", "quadratic"]

# Plot best-fitting model and binned data:
mp.modelfit(data, uncert, x, y, savefile="quad_bestfit.png")

# Plot trace plot:
mp.trace(posterior, zchain, pnames=pnames, savefile="quad_trace.png")

# Plot pairwise posteriors:
mp.pairwise(posterior, pnames=pnames, bestp=bestp, savefile="quad_pairwise.png")

# Plot marginal posterior histograms (with 68% highest-posterior-density credible regions):
mp.histogram(posterior, pnames=pnames, bestp=bestp, quantile=0.683,
    savefile="quad_hist.png")
