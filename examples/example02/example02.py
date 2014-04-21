# This script shows: (1) how to run MCMC from the shell, and (2) run MCMC 
# using multiple CPUs.

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# (This script assumes that you are in: .../demc/example/example02/)
# First, to see the full list of available arguments, run from the shell:
../../src/mcmc.py --help

# Arguments can be passes from the command line, from a configuration 
# file (recommended), or combining both.  Command line arguments have
# limitation (setting arrays, most probably, wont work).

# For example, check out the example configuration file:
config_demc.cfg
# Arguments are set by providing the keyword, followed by an equal sign,
# and then the value(s).  Arrays can be set by listing white-space (or
# new-line) separated values.  Strings don't need quotation marks around them.

# If an argument contains too many elements to list them in the configuration
# file, you can provide a file with the values by setting the path in the
# configuration file.  

# The files must contain one value per line (except for indparams).

# For simplicity, the 'params' file can hold other arguments as well.
# Simply include white-space-separated columns of values.
# For example a file can look like this:
#   par0 pmin0 pmax0 step0
#   par1 pmin1 pmax1 step1
#   par2 pmin2 pmax2 step2
#   ...

# The file can contain as few or as many columns provided that each line
# has the same number of columns and they are in this specific order:
#   params, pmin, pmax, stepsize, prior, priorlow, priorup

# The same goes for the 'data' file, it can also contain the uncertainties
# in a second column.

# The 'indparams' file will be interpreted as a list of values, one for
# each line. If there are more than one element per line, it will be
# interpreted as an array.  Obviously, in this case, the lines can have
# different number of elements.

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Part 1: We will run example01 again, this time from the shell: 

# First lets generate a dataset, and store the values in files to be read
# later from a configuration file:
import sys
import numpy as np
sys.path.append("../example01/")
from quadratic import quad

x  = np.linspace(0, 10, 100)  # Independent model variable
p0 = 3, -2.4, 0.5             # True-underlying model parameters
y  = quad(p0, x)              # Noiseless model
uncert = np.sqrt(np.abs(y))           # Data points uncertainty
error  = np.random.normal(0, uncert)  # Noise for the data
data   = y + error                    # Noisy data set

# Store the data and uncertainties in a file:
ndata = len(data)
dfile = "data_ex02.dat"
f = open(dfile, "w")
f.write("# Data   Uncert\n")
for i in np.arange(ndata): # One data, uncert value per line
    f.write("%10.3e  %10.3e\n"%(data[i], uncert[i]))

f.close()
# Store the independent parameters:
ifile = "indp_ex02.dat"
f = open(ifile, "w")
for i in np.arange(ndata): # Values separated by a white space
    f.write("%10.3e  "%(x[i]))

f.close()
# Store params, the boundaries, and the stepsize:
params   = np.array([ 20.0,  -2.0,   0.1])
pmin     = np.array([-10.0, -20.0, -10.0])
pmax     = np.array([ 40.0,  20.0,  10.0])
stepsize = np.array([  1.0,   0.5,   0.1])
nparams = len(params)
pfile = "ex01_pars.txt"
f = open(pfile, "w")
f.write("# param     pmin        pmax        stepsize\n")
for i in np.arange(nparams):
    f.write("%10.3e  %10.3e  %10.3e  %10.3e\n"%(params[i], pmin[i],
                                                pmax[i],   stepsize[i]))

f.close()

# Check again the provided configuration file, this is already setup to
# read these files, so now to run MCMC, run from the shell:
../../src/mcmc.py -c config_demc.cfg

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Part 2: Run MCMC in multiple processors:
# The module mpmc.py (multi-processor MCMC) wraps around mcmc.py to enable
# mutiprocessor capacity (using MPI).  It will use one CPU per MCMC-chain
# to calculate the model for the set of parameters in that chain.

# To run MCMC under MPI set the argument mpi to true in the config file:
#    'mpi     = True'
# and call the mpmc module from the shell:
../../src/mpmc.py -c config_demc.cfg
