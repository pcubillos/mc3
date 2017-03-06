#! /usr/bin/env python

# This script generates input files used to run MCMC from the shell prompt.

# Preamble
# --------
# To correctly execute this script, one needs to set the correct paths
# to the source code.  The paths are given as if the Python session
# runs from a 'run/' folder at the same level than the repo, as in:
#  rootdir/
#  |-- MCcubed/
#  `-- run/

# Alternatively, edit the paths from this script to adjust to your
# working directory.

# Import the necessary modules:
import sys
import numpy as np

# Import the modules from the MCcubed package:
sys.path.append("../MCcubed/")
import MCcubed as mc3

# Import the modeling function:
sys.path.append("../MCcubed/examples/models/")
from quadratic import quad


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
