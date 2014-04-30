#! /usr/bin/env python
import sys, os, subprocess
import argparse, ConfigParser
import timeit
import numpy as np
from mpi4py import MPI

import mcutils as mu
start = timeit.default_timer()

def main():
  """
  Multi-Processor Markov-Chain Monte Carlo (MPMC)

  This code calls MCMC to work under an MPI multiprocessor protocol or
  single-thread mode.  When using MPI it will launch one CPU per MCMC chain
  to work in parallel.

  Modification History:
  ---------------------
  2014-04-19  patricio  Initial implementation.  pcubillos@fulbrightmail.org
  """
  # Parse arguments:
  cparser = argparse.ArgumentParser(description=__doc__, add_help=False,
                         formatter_class=argparse.RawDescriptionHelpFormatter)
  # Add config file option:
  cparser.add_argument("-c", "--config_file",
                       help="Specify config file", metavar="FILE")
  # Remaining_argv contains all other command-line-arguments:
  args, remaining_argv = cparser.parse_known_args()

  # Get parameters from configuration file (if exists):
  cfile = args.config_file # The configuration file
  if not os.path.isfile(cfile):
    mu.exit(None, message="Configuration file: '%s' not found."%cfile)
  if cfile:
    config = ConfigParser.SafeConfigParser()
    config.read([cfile])
    defaults = dict(config.items("MCMC"))
  else:
    defaults = {}
  parser = argparse.ArgumentParser(parents=[cparser], add_help=False)
  parser.add_argument("-x", "--nchains",   dest="nchains", type=int,
                                           action="store", default=10)
  parser.add_argument(       "--mpi",      dest="mpi",     type=eval,
                    help="Run under MPI multiprocessing [default %(default)s]",
                     action="store", default=False)
  parser.add_argument("-T", "--tracktime", dest="tractime", action="store_true")
  parser.add_argument("-h", "--help",      dest="help", action="store_true")

  parser.set_defaults(**defaults)
  args2, unknown = parser.parse_known_args(remaining_argv)


  # The number of processors is the number of chains:
  nprocs    = args2.nchains
  mpi       = args2.mpi
  # Hidden feature to track the execution of the time per loop for MPI:
  tracktime = args2.tractime

  # Get source dir:
  iright = sys.argv[0].rfind('/')
  if iright == -1:
    sdir = "."
  else:
    sdir = sys.argv[0][:iright]

  # If asked for help:
  if args2.help:
    subprocess.call([sdir + "/mcmc.py --help"], shell=True)
    sys.exit(0)

  if not mpi:
    subprocess.call([sdir+"/mcmc.py " + " ".join(sys.argv[1:])], shell=True)
  else:
    if tracktime:
      start_mpi = timeit.default_timer()
    # Call MCMC:
    args = [sdir+"/mcmc.py", "-c"+cfile] + remaining_argv
    comm1 = MPI.COMM_SELF.Spawn(sys.executable, args=args, maxprocs=1)

    # Get OK flag from MCMC:
    abort = np.array([0])
    mu.comm_gather(comm1, abort)
    if abort[0]:
      mu.exit(comm1)

    # Call wrapper of model function:
    args = [sdir+"/func.py", "-c"+cfile] + remaining_argv
    comm2 = MPI.COMM_SELF.Spawn(sys.executable, args=args, maxprocs=nprocs)

    # MPI get sizes from MCMC:
    array1 = np.zeros(3, np.int)
    mu.comm_gather(comm1, array1)
    npars, ndata, niter = array1
    # MPI Broadcast to workers:
    mu.comm_bcast(comm2, np.asarray([npars, niter], np.int), MPI.INT)

    # get npars, ndata from MCMC:
    mpipars   = np.zeros(npars*nprocs, np.double)
    mpimodels = np.zeros(ndata*nprocs, np.double)

    if tracktime:
      start_loop = timeit.default_timer()
      loop_timer = []
      loop_timer2 = []
    while niter >= 0:
      if tracktime:
        loop_timer.append(timeit.default_timer())
      # Gather (receive) parameters from MCMC:
      mu.comm_gather(comm1, mpipars)

      # Scatter (send) parameters to funcwrapper:
      mu.comm_scatter(comm2, mpipars, MPI.DOUBLE)
      # Gather (receive) models:
      mu.comm_gather(comm2, mpimodels)

      # Scatter (send) results to MCMC:
      mu.comm_scatter(comm1, mpimodels, MPI.DOUBLE)
      niter -= 1
      if tracktime:
        loop_timer2.append(timeit.default_timer() - loop_timer[-1])

    if tracktime:
      stop = timeit.default_timer()

    if mpi:
      comm1.barrier()
      comm2.barrier()
      comm1.Disconnect()
      comm2.Disconnect()

    #if bench == True:
    if tracktime:
      print("Total execution time:   %10.6f s"%(stop - start))
      print("Time to initialize MPI: %10.6f s"%(start_loop - start_mpi))
      print("Time to run first loop: %10.6f s"%(loop_timer[1] - loop_timer[0]))
      print("Time to run last loop:  %10.6f s"%(loop_timer[-1]- loop_timer[-2]))
      print("Time to run avg loop:   %10.6f s"%(np.mean(loop_timer2)))

    # Close communications and disconnect: 
    #mu.exit(comm1, comm2=comm2)


if __name__ == "__main__":
  main()
