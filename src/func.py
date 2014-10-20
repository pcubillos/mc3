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
  2014-06-25  patricio  Added support for inner-MPI loop.
  """
  # Parse arguments:
  cparser = argparse.ArgumentParser(description=__doc__, add_help=False,
                         formatter_class=argparse.RawDescriptionHelpFormatter)
  # Add config file option:
  cparser.add_argument("-c", "--config_file", 
                       help="Configuration file", metavar="FILE")
  # Remaining_argv contains all other command-line-arguments:
  args, remaining_argv = cparser.parse_known_args()

  # Get parameters from configuration file:
  cfile = args.config_file
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

  # Add path to func:
  if len(args2.func) == 3:
    sys.path.append(args2.func[2])

  rank = comm.Get_rank()
  verb = rank == 0

  # If func is an MPI loop of functions:
  if args2.func[0] == "main":
    #print("FUNC FLAG 16: func is main (rank: %d/%d)"%(comm.Get_rank(),
    #                                                  comm.Get_size()))
    exec('import {:s} as funcmodule'.format(args2.func[1]))
    # Get Sub-routines and func name:
    subfuncs, func = funcmodule.main()
    # Initialize the MPI communicators with subroutines:
    comms = []
    if rank != 0:
      rargs = ["--quiet"]
    else:
      rargs = []
    for subfunc in subfuncs:
      # Get config file for non-Python ssub-functions:
      subcfile = cfile
      if not subfunc.endswith(".py"):
        subcfile = os.getcwd() + "/" + dict(config.items("MCMC"))["config"]
      comms.append(mu.comm_spawn(subfunc, 1, subcfile, rargs=rargs,
                                 path=args2.func[2]))
  else:
    exec('from {:s} import {:s} as func'.format(args2.func[1], args2.func[0]))
    # Get indparams from configuration file:
    if args2.indparams != [] and os.path.isfile(args2.indparams[0]):
      indparams = mu.readbin(args2.indparams[0])


  # Get the number of parameters and iterations from MPI:
  array1 = np.zeros(2, np.int)
  mu.comm_bcast(comm, array1)
  npars, niter = array1
  #mu.msg(verb, "FUNC FLAG 30: npar={:d}, niter={:d}".format(npars, niter))

  # Initialization:
  if args2.func[0] == "main":
    initargs = comms + [npars, niter] + [rank == 0]
    commarrays = funcmodule.init(*initargs)
    indparams = comms + commarrays + [rank == 0]
    #mu.msg(verb, "FUNC FLAG 35: sizes = {}".format([np.size(commarrays[0]),
    #                        np.size(commarrays[1]), np.size(commarrays[2])]))

  # Allocate array to receive parameters from MPI:
  params = np.zeros(npars, np.double)

  # Main MCMC Loop:
  while niter >= 0:
    # Receive parameters from MCMC:
    mu.comm_scatter(comm, params)

    # Evaluate model:
    fargs = [params] + indparams  # List of function's arguments
    model = func(*fargs)

    # Send resutls:
    mu.comm_gather(comm, model, MPI.DOUBLE)
    niter -= 1

  # Close inner-loop communicators:
  if args2.func[0] == "main":
    for c in comms:
      c.Barrier()
      c.Disconnect()

  #mu.msg(verb, "FUNC FLAG 99: func out")
  # Close communications and disconnect:
  mu.exit(comm)


if __name__ == "__main__":
  # Open communications with the master:
  comm = MPI.Comm.Get_parent()
  main(comm)
