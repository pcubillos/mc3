# This script shows how to run MCMC from the shell.

# This script assumes that your current folder is /example/example02/

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# First, to see the full list of available arguments, run from the shell:
../../src/mccubed.py --help

# Arguments can be passes from the command line or from a configuration 
# file (recommended), or combining both.

# For example, check out the example configuration file:
#  config_demc.cfg
# Arguments are set by providing the keyword, followed by an equal sign,
# and then the value(s).  Arrays can be set by listing empty-space (or
# new-line) separated values.  Strings don't need quotation marks around them.

# This example uses these data and params files created as in the example01:
#  data_ex02.dat
#  pars_ex02.dat
#  indp_ex02.dat

# To run MCMC, execute from the terminal/bash/shell:
mpirun ../../src/mccubed.py -c config_demc.cfg

# Command-line arguments override configuration-file arguments.
