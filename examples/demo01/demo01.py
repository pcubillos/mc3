# Preamble:
# --------

# To run this script, cd into the folder that contains the repository.
# (i.e., if you do 'ls', you will see the 'MCcubed/' folder).
# Make and cd into a 'run/' folder to run this demo, i.e.:
# $ mkdir run
# $ cd run
# And start an interactive Python session.

# Alternatively, edit the paths from this script to adjust to your
# working directory.


# demo01.py:

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../MCcubed/")
import MCcubed as mc3

# Get function to model (and sample):
sys.path.append("../MCcubed/examples/models/")
from quadratic import quad

# Create a synthetic dataset:
x = np.linspace(0, 10, 100)          # Independent model variable
p0 = 3, -2.4, 0.5                    # True-underlying model parameters
y = quad(p0, x)                      # Noiseless model
uncert = np.sqrt(np.abs(y))          # Data points uncertainty
error = np.random.normal(0, uncert)  # Noise for the data
data = y + error                     # Noisy data set

# Fit the quad polynomial coefficients:
params = np.array([ 20.0, -2.0, 0.1])  # Initial guess of fitting params.

# Run the MCMC:
posterior, bestp = mc3.mcmc(data, uncert, func=quad, indparams=[x],
                            params=params, numit=3e4, burnin=100)
