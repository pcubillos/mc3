# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = [
    'ROOT',
    'parray',
    'saveascii', 'loadascii', 'savebin', 'loadbin',
    'isfile',
    'binarray', 'weightedbin',
    'credregion',
    'default_parnames',
    ]

import os
import sys

import numpy as np
import scipy.stats as stats
import scipy.interpolate as si

ROOT = os.path.realpath(os.path.dirname(__file__) + '/../..') + '/'
sys.path.append(ROOT + 'MCcubed/lib/')
from binarray import binarray, weightedbin

if sys.version_info.major == 2:
    range = xrange


def parray(string):
    """
    Convert a string containin a list of white-space-separated (and/or
    newline-separated) values into a numpy array
    """
    if string == 'None':
        return None
    try:    # If they can be converted into doubles, do it:
        return np.asarray(string.split(), np.double)
    except: # Else, return a string array:
        return string.split()


def saveascii(data, filename, precision=8):
    """
    Write (numeric) data to ASCII file.

    Parameters
    ----------
    data:  1D/2D numeric iterable (ndarray, list, tuple, or combination)
        Data to be stored in file.
    filename:  String
        File where to store the arrlist.
    precision: Integer
        Maximum number of significant digits of values.

    Example
    -------
    >>> import numpy as np
    >>> import MCcubed.utils as mu

    >>> a = np.arange(4) * np.pi
    >>> b = np.arange(4)
    >>> c = np.logspace(0, 12, 4)

    >>> outfile = 'delete.me'
    >>> mu.saveascii([a,b,c], outfile)

    >>> # This will produce this file:
    >>> with open(outfile) as f:
    >>>   print(f.read())
            0         0         1
    3.1415927         1     10000
    6.2831853         2     1e+08
     9.424778         3     1e+12
    """
    # Force it to be a 2D ndarray:
    data = np.array(data, ndmin=2).T

    # Save arrays to ASCII file:
    with open(filename, 'w') as f:
        for parvals in data:
            f.write(' '.join('{:9.{:d}g}'.format(v,precision)
                    for v in parvals) + '\n')


def loadascii(filename):
    """
    Extract data from file and store in a 2D ndarray (or list of arrays
    if not square).  Blank or comment lines are ignored.

    Parameters
    ----------
    filename: String
        Name of file containing the data to read.

    Returns
    -------
    array: 2D ndarray or list
        See parameters description.
    """
    # Open and read the file:
    lines = []
    for line in open(filename, 'r'):
        if not line.startswith('#') and line.strip() != '':
            lines.append(line)

    # Count number of lines:
    npars = len(lines)

    # Extract values:
    ncolumns = len(lines[0].split())
    array = np.zeros((npars, ncolumns), np.double)
    for i, line in enumerate(lines):
        array[i] = line.strip().split()
    array = np.transpose(array)

    return array


def savebin(data, filename):
    """
    Write data variables into a numpy npz file.

    Parameters
    ----------
    data:  List of data objects
        Data to be stored in file.  Each array must have the same length.
    filename:  String
        File where to store the arrlist.

    Note
    ----
    This wrapper around np.savez() preserves the data type of list and
    tuple variables when the file is open with loadbin().

    Example
    -------
    >>> import MCcubed.utils as mu
    >>> import numpy as np
    >>> # Save list of data variables to file:
    >>> datafile = 'datafile.npz'
    >>> indata = [np.arange(4), 'one', np.ones((2,2)), True, [42], (42, 42)]
    >>> mu.savebin(indata, datafile)
    >>> # Now load the file:
    >>> outdata = mu.loadbin(datafile)
    >>> for data in outdata:
    >>>     print(repr(data))
    array([0, 1, 2, 3])
    'one'
    array([[ 1.,  1.],
           [ 1.,  1.]])
    True
    [42]
    (42, 42)
    """
    # Get the number of elements to determine the key's fmt:
    ndata = len(data)
    fmt = len(str(ndata))

    key = []
    for i, datum in enumerate(data):
        dkey = 'file{:{}d}'.format(i, fmt)
        # Encode in the key if a variable is a list or tuple:
        if isinstance(datum, list):
            dkey += '_list'
        elif isinstance(datum, tuple):
            dkey += '_tuple'
        elif isinstance(datum, str):
            dkey += '_str'
        elif isinstance(datum, bool):
            dkey += '_bool'
        key.append(dkey)

    # Use a dictionary so savez() include the keys for each item:
    d = dict(zip(key, data))
    np.savez(filename, **d)


def loadbin(filename):
    """
    Read a binary npz array, casting list and tuple variables into
    their original data types.

    Parameters
    ----------
    filename: String
       Path to file containing the data to be read.

    Return
    ------
    data:  List
       List of objects stored in the file.

    Example
    -------
    See example in savebin().
    """
    # Unpack data:
    npz = np.load(filename)
    data = []
    for key, val in sorted(npz.items()):
        data.append(val[()])
        # Check if val is a str, bool, list, or tuple:
        if '_' in key:
            exec('data[-1] = ' + key[key.find('_')+1:] + '(data[-1])')

    return data


def isfile(input, iname, log, dtype, unpack=True, not_none=False):
    """
    Check if an input is a file name; if it is, read it.
    Genereate error messages if it is the case.

    Parameters
    ----------
    input: Iterable or String
        The input variable.
    iname: String
        Input-variable name.
    log: File pointer
         If not None, print message to the given file pointer.
    dtype: String
        File data type, choose between 'bin' or 'ascii'.
    unpack: Bool
        If True, return the first element of a read file.
    not_none: Bool
        If True, throw an error if input is None.
    """
    # Set the loading function depending on the data type:
    if dtype == 'bin':
        load = loadbin
    elif dtype == 'ascii':
        load = loadascii
    else:
        log.error("Invalid data type '{:s}', must be either 'bin' or 'ascii'.".
                  format(dtype), tracklev=-3)

    # Check if the input is None, throw error if requested:
    if input is None:
        if not_none:
            log.error("'{:s}' is a required argument.".format(iname),
                      tracklev=-3)
        return None

    # Check that it is an iterable:
    if not np.iterable(input):
        log.error('{:s} must be an iterable or a file name.'.format(iname),
                  tracklev=-3)

    # Check if it is a string, a string in a list, or an array:
    if isinstance(input, str):
        ifile = input
    elif isinstance(input[0], str):
        ifile = input[0]
    else:
        return input

    # It is a file name:
    if not os.path.isfile(ifile):
        log.error("{:s} file '{:s}' not found.".format(iname, ifile),
                  tracklev=-3)
    if unpack:  # Unpack (remove outer dimension) if necessary
        return load(ifile)[0]
    return load(ifile)


def credregion(posterior=None, percentile=0.6827, pdf=None, xpdf=None):
    """
    Compute a smoothed posterior density distribution and the minimum
    density for a given percentile of the highest posterior density.

    These outputs can be used to easily compute the HPD credible regions.

    Parameters
    ----------
    posterior: 1D float ndarray
        A posterior distribution.
    percentile: Float
        The percentile (actually the fraction) of the credible region.
        A value in the range: (0, 1).
    pdf: 1D float ndarray
        A smoothed-interpolated PDF of the posterior distribution.
    xpdf: 1D float ndarray
        The X location of the pdf values.

    Returns
    -------
    pdf: 1D float ndarray
        A smoothed-interpolated PDF of the posterior distribution.
    xpdf: 1D float ndarray
        The X location of the pdf values.
    HPDmin: Float
        The minimum density in the percentile-HPD region.

    Example
    -------
    >>> import numpy as np
    >>> import MCcubed.utils as mu
    >>> # Test for a Normal distribution:
    >>> npoints = 100000
    >>> posterior = np.random.normal(0, 1.0, npoints)
    >>> pdf, xpdf, HPDmin = mu.credregion(posterior)
    >>> # 68% HPD credible-region boundaries (somewhere close to +/-1.0):
    >>> print(np.amin(xpdf[pdf>HPDmin]), np.amax(xpdf[pdf>HPDmin]))

    >>> # Re-compute HPD for the 95% (withour recomputing the PDF):
    >>> pdf, xpdf, HPDmin = mu.credregion(pdf=pdf, xpdf=xpdf, percentile=0.9545)
    >>> print(np.amin(xpdf[pdf>HPDmin]), np.amax(xpdf[pdf>HPDmin]))
    """
    if pdf is None and xpdf is None:
        # Thin if posterior has too many samples (> 120k):
        thinning = np.amax([1, int(np.size(posterior)/120000)])
        # Compute the posterior's PDF:
        kernel = stats.gaussian_kde(posterior[::thinning])
        # Remove outliers:
        mean = np.mean(posterior)
        std  = np.std(posterior)
        k = 6
        lo = np.amax([mean-k*std, np.amin(posterior)])
        hi = np.amin([mean+k*std, np.amax(posterior)])
        # Use a Gaussian kernel density estimate to trace the PDF:
        x  = np.linspace(lo, hi, 100)
        # Interpolate-resample over finer grid (because kernel.evaluate
        #  is expensive):
        f    = si.interp1d(x, kernel.evaluate(x))
        xpdf = np.linspace(lo, hi, 3000)
        pdf  = f(xpdf)

    # Sort the PDF in descending order:
    ip = np.argsort(pdf)[::-1]
    # Sorted CDF:
    cdf = np.cumsum(pdf[ip])
    # Indices of the highest posterior density:
    iHPD = np.where(cdf >= percentile*cdf[-1])[0][0]
    # Minimum density in the HPD region:
    HPDmin = np.amin(pdf[ip][0:iHPD])
    return pdf, xpdf, HPDmin


def default_parnames(npars):
    """
    Create an array of parameter names with sequential indices.

    Parameters
    ----------
    npars: Integer
        Number of parameters.

    Results
    -------
    1D string ndarray of parameter names.
    """
    namelen = len(str(npars))
    return np.array(['Param {:0{}d}'.format(i+1,namelen) for i in range(npars)])
