# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

import sys
import os
import warnings
if sys.version_info.major == 3:
    import configparser
else:
    import ConfigParser as configparser

import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt

import MCcubed as mc3


def main():
    """
    Multi-Core Markov-chain Monte Carlo (MC3) wrapper for the command-line
    (shell) call.

    Notes
    -----
    To display the full list of arguments, run from the prompt:
    mc3 -h

    Command-line arguments overwrite the config-file arguments if the
    argument is defined twice.
    """
    parser = mc3.mc.parse()

    # Parse command-line args (right now, just interested in the config file):
    args, unknown = parser.parse_known_args()

    # Parse configuration file to a dictionary:
    if args.cfile is not None and not os.path.isfile(args.cfile):
        print("Configuration file: '{:s}' not found.".format(args.cfile))
        sys.exit(0)
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
    mc3.mcmc(**vars(args))


if __name__ == "__main__":
    plt.ioff()
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
