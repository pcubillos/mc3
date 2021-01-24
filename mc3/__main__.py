# Copyright (c) 2015-2021 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

import sys
import os
import warnings
import argparse
import configparser

import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt

import mc3
import mc3.utils as mu


def main():
    """
    Multi-Core Markov-chain Monte Carlo (MC3) wrapper for the command line

    Notes
    -----
    To display the full list of arguments, run from the prompt:
    mc3 -h

    Command-line arguments overwrite the config-file arguments if the
    argument is defined twice.
    """
    parser = parse()

    # Parse command-line args (right now, just interested in the config file):
    args, unknown = parser.parse_known_args()

    # Parse configuration file to a dictionary:
    if args.cfile is not None and not os.path.isfile(args.cfile):
        print("Configuration file: '{:s}' not found.".format(args.cfile))
        sys.exit(0)
    if args.cfile:
        config = configparser.ConfigParser()
        config.read([args.cfile])
        defaults = dict(config.items("MCMC"))
    else:
        defaults = {}
    # Set defaults from the configuration-file values:
    parser.set_defaults(**defaults)
    # Overwrite defaults with the command-line arguments:
    args, unknown = parser.parse_known_args()
    delattr(args, 'cfile')
    # Call MCMC driver:
    mc3.sample(**vars(args))


def parse():
    """
    MC3 command-line argument parser.
    """
    # Parse the config file from the command line:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Configuration-file option:
    parser.add_argument("-c", "--cfile",
        help="Configuration file.", metavar="FILE")
    parser.add_argument("-v", "--version", action="version",
                       help="Show MC3's version.",
                       version='MC3 version {:s}.'.format(mc3.__version__))
    # MCMC Options:
    group = parser.add_argument_group("MCMC General Options")
    group.add_argument("--nsamples",  dest="nsamples", action="store",
        type=eval, default=None,
        help="Number of MCMC samples.")
    group.add_argument("--nchains",   dest="nchains", action="store",
        type=int,  default=7,
        help="Number of chains [default: %(default)s]")
    group.add_argument("--ncpu",     dest="ncpu",   action="store",
        type=int,  default=None,
        help="Number of CPUs for the chains [default: nchains+1]")
    group.add_argument("--sampler",  dest="sampler", action="store",
        type=str,  default=None,
        help="Sampler algorithm, select from: ['mrw', 'demc', 'snooker'].")
    group.add_argument("--wlike",     dest="wlike", action="store",
        type=eval, default=False,
        help="Calculate the likelihood in a wavelet base "
             "[default: %(default)s]")
    group.add_argument("--leastsq",   dest="leastsq", action="store",
        type=eval, default=None,
        help="If not None, perform a least-squares optimiztion before the "
             "MCMC run.  Select from: 'lm': Levenberg-Marquardt (most "
             "efficient, but does not obey bounds); 'trf': Trust Region "
             "Reflective [default: %(default)s].")
    group.add_argument("--chisqscale", dest="chisqscale", action="store",
        type=eval, default=False,
        help="Scale the data uncertainties such that the reduced "
             "chi-squared = 1. [default: %(default)s]")
    group.add_argument("--grtest",     dest="grtest", action="store",
        type=eval, default=False,
        help="Run Gelman-Rubin test [default: %(default)s]")
    group.add_argument("--grbreak",   dest="grbreak", action="store",
        type=float, default=0.0,
        help="Gelman-Rubin convergence threshold to stop the MCMC.  I'd "
             "suggest grbreak ~ 1.001 -- 1.005.  Do not break if "
             "grbreak=0.0 (default).")
    group.add_argument("--grnmin",     dest="grnmin", action="store",
        type=eval, default=0.5,
        help="Minimum number of valid samples required for grbreak.  If grnmin "
             "is integer, require at least grnmin samples to break out of the "
             "MCMC. If grnmin is a float (in the range 0.0--1.0), require at "
             "least grnmin * maximum number of samples to break out of the "
             "MCMC [default: %(default)s]")
    group.add_argument("--burnin",    dest="burnin", action="store",
        type=eval, default=0,
        help="Number of burn-in iterations (per chain) [default: %(default)s]")
    group.add_argument("--thinning",  dest="thinning", action="store",
        type=int,  default=1,
        help="Chains thinning factor (use every thinning-th iteration) for GR "
             "test and plots [default: %(default)s]")
    group.add_argument("--fgamma",    dest="fgamma",   action="store",
        type=float, default=1.0,
        help="Scaling factor for DEMC's gamma [default: %(default)s]")
    group.add_argument("--fepsilon",  dest="fepsilon", action="store",
        type=float, default=0.0,
        help="Scaling factor for DEMC's support distribution "
             "[default: %(default)s]")
    group.add_argument("--hsize",     dest="hsize", action="store",
        type=int,  default=10,
        help="Number of initial samples per chain [default: %(default)s]")
    group.add_argument("--kickoff",   dest="kickoff", action="store",
        type=str,  default="normal",
        help="Chain's starter mode, select between: ['normal', 'uniform']. "
             "[default: %(default)s]")
    group.add_argument("--plots",     dest="plots", action="store",
        type=eval, default=False,
        help="If True, generate output figures. [default: %(default)s]")
    group.add_argument("--ioff",     dest="ioff", action="store",
        type=eval, default=False,
        help="If True, set plt.ioff(), i.e., do not display figures on screen "
             "[default: %(default)s]")
    group.add_argument("--showbp",   dest="showbp", action="store",
        type=eval, default=True,
        help="If True, show best-fitting values in histogram and pairwise "
             "plots [default: %(default)s]")
    group.add_argument("--savefile", dest="savefile", action="store",
        type=str,  default=None,
        help="Output npz filename to store the parameter posterior "
             "distributions [default: %(default)s]")
    group.add_argument("-r", "--resume", dest="resume", action="store_true",
        default=False,
        help="If set, resume a previous run (load output).")
    group.add_argument("--rms",       dest="rms", action="store",
        type=eval, default=False,
        help="If True, calculate RMS-vs-binsize of data--bestmodel residuals "
             "[default: %(default)s]")
    group.add_argument("--log",       dest="log", action="store",
        type=str,  default=None,
        help="Output log filename.")
    group.add_argument("--pnames",   dest="pnames", action="store",
        type=mu.parray, default=None,
        help="List of parameter names for screen output (and figures if "
             "texnames is not defined).  If pnames is not defined, default "
             "to texnames.")
    group.add_argument("--texnames",    dest="texnames", action="store",
        type=mu.parray, default=None,
        help="List of parameter names for figures (may use latex syntax). "
             "[default: None]")
    # Fitting-parameter Options:
    group = parser.add_argument_group("Fitting-function Options")
    group.add_argument("--func",       dest="func", action="store",
        type=mu.parray, default=None,
        help="List of strings with the function name, module "
             "name, and path-to-module [required]")
    group.add_argument("--params",     dest="params", action="store",
        type=mu.parray, default=None,
        help="Filename or list of initial-guess model-fitting "
             "parameter [required]")
    group.add_argument("--pmin",       dest="pmin", action="store",
        type=mu.parray, default=None,
        help="Filename or list of parameter lower boundaries "
             "[default: -inf for each parameter]")
    group.add_argument("--pmax",       dest="pmax", action="store",
        type=mu.parray, default=None,
        help="Filename or list of parameter upper boundaries "
             "[default: +inf for each parameter]")
    group.add_argument("--pstep",   dest="pstep", action="store",
        type=mu.parray, default=None,
        help="Parameter stepping [required].  Additionally, parameters with "
             "pstep=0 are fixed, parameters with negative pstep are "
             "shared (see documentation).")
    group.add_argument("--indparams",  dest="indparams", action="store",
        type=mu.parray, default=[],
        help="Filename or list with independent parameters for func "
             "[default: None]")
    # Data Options:
    group = parser.add_argument_group("Data Options")
    group.add_argument("--data",     dest="data", action="store",
        type=mu.parray,    default=None,
        help="Filename or array of the data being fitted [required]")
    group.add_argument("--uncert",   dest="uncert", action="store",
        type=mu.parray,    default=None,
        help="Filemane or array with the data uncertainties [required]")
    group.add_argument("--prior",    dest="prior", action="store",
        type=mu.parray,    default=None,
        help="Filename or array with parameter prior estimates "
             "[default: %(default)s]")
    group.add_argument("--priorlow", dest="priorlow", action="store",
        type=mu.parray,    default=None,
        help="Filename or array with prior lower uncertainties "
             "[default: %(default)s]")
    group.add_argument("--priorup",  dest="priorup", action="store",
        type=mu.parray,    default=None,
        help="Filename or array with prior upper uncertainties "
             "[default: %(default)s]")
    return parser


if __name__ == "__main__":
    plt.ioff()
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
