#! /usr/bin/env python

# To run this script, cd into the folder that contains the repository
# (i.e., if you do 'ls', you will see the 'MCcubed/' folder).
# Make and cd into a 'demo02/' folder to run this demo, i.e.:
# $ mkdir demo02
# $ cd demo02
# Alternatively, edit the paths below.


# Import the necessary modules:
import sys
import numpy as np

# Import the modules from the MCcubed package:
sys.path.append("../MCcubed/")
import MCcubed as mc3
sys.path.append("../MCcubed/examples/models/")
from quadratic import quad


# Create a synthetic dataset using a quadratic polynomial curve:
x  = np.linspace(0.0, 10, 100)        # Independent model variable
p0 = 3, -2.4, 0.5                     # True-underlying model parameters
y  = quad(p0, x)                      # Noiseless model
uncert = np.sqrt(np.abs(y))           # Data points uncertainty
error  = np.random.normal(0, uncert)  # Noise for the data
data   = y + error                    # Noisy data set

# data.npz contains the data and uncertainty arrays:
mc3.utils.savebin([data, uncert], 'data.npz')
# indp.npz contains a list of variables:
mc3.utils.savebin([x],      'indp.npz')
