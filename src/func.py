#! /usr/bin/env python
import sys, os
import argparse, ConfigParser
import numpy as np
from mpi4py import MPI

import mcutils as mu

def main(comm):
  """
  Wrapper of modeling function for MCMC under MPI protocol.

  Modification History:
  ---------------------
  2014-04-19  patricio  Initial implementation.  pcubillos@fulbrightmail.org
  """
  # Parse arguments:
  cparser = argparse.ArgumentParser(description=__doc__, add_help=False,
                         formatter_class=argparse.RawDescriptionHelpFormatter)
  # Add config file option:
  cparser.add_argument("-c", "--config_file", 
                       help="Configuration file", metavar="FILE")
  # Remaining_argv contains all other command-line-arguments:
  args, remaining_argv = cparser.parse_known_args()

  # Get parameters from configuration file (if exists):
  cfile = args.config_file # The configuration file
  if cfile:
    config = ConfigParser.SafeConfigParser()
    config.read([cfile])
    defaults = dict(config.items("MCMC"))
  else:
    defaults = {}
  parser = argparse.ArgumentParser(parents=[cparser])
  parser.add_argument("-f", "--func",      dest="func",     type=mu.parray, 
                                           action="store",  default=None)
  parser.add_argument("-i", "--indparams", dest="indparams", type=mu.parray, 
                                           action="store",   default=[])
  parser.set_defaults(**defaults)
  args2, unknown = parser.parse_known_args(remaining_argv)

  # Get indparams from configuration file:
  if args2.indparams != [] and os.path.isfile(args2.indparams[0]):
    indparams = mu.read2array(args2.indparams[0], square=False)

  # Get func from configuration file:
  if len(args2.func) == 3:
    sys.path.append(args2.func[2])
    exec('from %s import %s as func'%(args2.func[1], args2.func[0]))

  # Get the number of parameters and iterations from MPI:
  array1 = np.zeros(2, np.int)
  mu.comm_bcast(comm, array1)
  npars, niter = array1

  # Allocate array to receive parameters from MPI:
  params = np.zeros(npars, np.double)
  while niter >= 0:
    # Receive parameters from master:
    mu.comm_scatter(comm, params)
    # Evaluate model:
    fargs = [params] + indparams  # List of function's arguments
    model = func(*fargs)
    # Send resutls:
    mu.comm_gather(comm, model, MPI.DOUBLE)
    niter -= 1

  # Close communications and disconnect:
  mu.exit(comm)


if __name__ == "__main__":
  # Open communications with the master:
  comm = MPI.Comm.Get_parent()
  main(comm)
