#! /usr/bin/env python

# Copyright (c) 2015-2018 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

import sys
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

# Config Parser changed between Python2 and Python3:
if sys.version_info.major == 3:
  import configparser
else:
  import ConfigParser as configparser

# Import MC3 package:
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import MCcubed as mc3
import MCcubed.utils as mu


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

  # Call MCMC driver:
  output = mc3.mcmc(**vars(args))


if __name__ == "__main__":
  plt.ioff()
  warnings.simplefilter("ignore", RuntimeWarning)
  main()
