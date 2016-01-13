#!/bin/bash

# To run this script, cd into the folder that contains the repository
# (i.e., if you do 'ls', you will see the 'MCcubed/' folder).
# Make and cd into a 'demo02/' folder to run this demo, i.e.:
# $ mkdir demo02
# $ cd demo02

# Alternatively, edit the paths in the 'MCMC.cfg' configuration file
# and the following line.

# Run MC3 from shell:
mpirun ../MCcubed/src/mccubed.py -c MCMC.cfg
