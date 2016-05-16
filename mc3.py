#! /usr/bin/env python

# Copyright (c) 2015-2016 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

import sys, os
import warnings
import argparse
import configparser
import numpy as np

# Import MC3 package:
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import MCcubed as mc3
mu = mc3.utils


def main():
  """
  Multi-Core Markov-chain Monte Carlo (MC3) wrapper for the command-line
  (shell) call.

  Notes
  -----
  1.- To display the full list of arguments, run from the prompt:
      mccubed.py -h
  2.- The command line overwrites over the config file in case an argument
      is defined twice.
  """

  parser = mc3.mc.parse()

  # Parse command-line args (right now, just interested in the config file):
  args, unknown = parser.parse_known_args()

  # Parse configuration file to a dictionary:
  if args.cfile is not None and not os.path.isfile(args.cfile):
    mu.error("Configuration file: '{:s}' not found.".format(args.cfile))
  if args.cfile:
    config = configparser.SafeConfigParser()
    config.read([args.cfile])
    defaults = dict(config.items("MCMC"))
  else:
    defaults = {}
  # Set defaults from the configuration-file values:
  parser.set_defaults(**defaults)
  # Overwrite defaults with the command-line arguments:
  args, unknown = parser.parse_known_args()

  # Unpack configuration-file/command-line arguments:
  for key in vars(args).keys():
    exec("{:s} = args.{:s}".format(key, key))

  # Call MCMC driver:
  output = mc3.mcmc(data, uncert, func, indparams,
                    params, pmin, pmax, stepsize,
                    prior, priorlow, priorup,
                    nsamples, nchains, walk, wlike,
                    leastsq, chisqscale, grtest, burnin,
                    thinning, hsize, kickoff,
                    plots, savefile, savemodel, resume,
                    rms, log)


if __name__ == "__main__":
  warnings.simplefilter("ignore", RuntimeWarning)
  main()

