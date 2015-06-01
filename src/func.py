#! /usr/bin/env python

# ******************************* START LICENSE *****************************
# 
# Multi-Core Markov-chain Monte Carlo (MC3), a code to estimate
# model-parameter best-fitting values and Bayesian posterior
# distributions.
# 
# This project was completed with the support of the NASA Planetary
# Atmospheres Program, grant NNX12AI69G, held by Principal Investigator
# Joseph Harrington.  Principal developers included graduate student
# Patricio E. Cubillos and programmer Madison Stemm.  Statistical advice
# came from Thomas J. Loredo and Nate B. Lust.
# 
# Copyright (C) 2014 University of Central Florida.  All rights reserved.
# 
# This is a test version only, and may not be redistributed to any third
# party.  Please refer such requests to us.  This program is distributed
# in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.
# 
# Our intent is to release this software under an open-source,
# reproducible-research license, once the code is mature and the first
# research paper describing the code has been accepted for publication
# in a peer-reviewed journal.  We are committed to development in the
# open, and have posted this code on github.com so that others can test
# it and give us feedback.  However, until its first publication and
# first stable release, we do not permit others to redistribute the code
# in either original or modified form, nor to publish work based in
# whole or in part on the output of this code.  By downloading, running,
# or modifying this code, you agree to these conditions.  We do
# encourage sharing any modifications with us and discussing them
# openly.
# 
# We welcome your feedback, but do not guarantee support.  Please send
# feedback or inquiries to:
# 
# Joseph Harrington <jh@physics.ucf.edu>
# Patricio Cubillos <pcubillos@fulbrightmail.org>
# 
# or alternatively,
# 
# Joseph Harrington and Patricio Cubillos
# UCF PSB 441
# 4111 Libra Drive
# Orlando, FL 32816-2385
# USA
# 
# Thank you for using MC3!
# ******************************* END LICENSE *******************************

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
  2014-10-23  patricio  Removed inner-MPI loop.
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

  exec('from {:s} import {:s} as func'.format(args2.func[1], args2.func[0]))
  # Get indparams from configuration file:
  if args2.indparams != [] and os.path.isfile(args2.indparams[0]):
    indparams = mu.readbin(args2.indparams[0])


  # Get the number of parameters and iterations from MPI:
  array1 = np.zeros(2, np.int)
  mu.comm_bcast(comm, array1)
  npars, niter = array1

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

  # Close communications and disconnect:
  mu.comm_disconnect(comm)

if __name__ == "__main__":
  # Open communications with the master:
  comm = MPI.Comm.Get_parent()
  main(comm)
