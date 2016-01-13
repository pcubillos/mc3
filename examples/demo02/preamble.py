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
sys.path.append("../../src")
import mcutils as mu
sys.path.append("../models/")
from quadratic import quad


# Create a synthetic dataset using a quadratic polynomial curve:
x  = np.linspace(0, 10, 100)          # Independent model variable
p0 = 3, -2.4, 0.5                     # True-underlying model parameters
y  = quad(p0, x)                      # Noiseless model
uncert = np.sqrt(np.abs(y))           # Data points uncertainty
error  = np.random.normal(0, uncert)  # Noise for the data
data   = y + error                    # Noisy data set

mu.savebin([data, uncert], 'data.npz')
# indparams contains additional arguments of func (if necessary). Each
# additional argument is an item in the indparams tuple:
mu.savebin([x],      'indp.npz')
