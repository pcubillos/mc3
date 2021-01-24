# Copyright (c) 2015-2021 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = [
    'ROOT',
    'ignore_system_exit',
    'parray',
    'saveascii', 'loadascii',
    'savebin', 'loadbin',
    'isfile',
    'burn',
    'default_parnames',
    ]

import os
import sys
import functools

import numpy as np

from .log import Log

ROOT = os.path.realpath(os.path.dirname(__file__) + '/../..') + '/'


def ignore_system_exit(func):
    """Decorator to ignore SystemExit exceptions."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SystemExit:
            return None
    return new_func


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
    >>> import mc3.utils as mu

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
    >>> import mc3.utils as mu
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


def burn(Zdict=None, burnin=None, Z=None, zchain=None, sort=True):
    """
    Return a posterior distribution removing the burnin initial iterations
    of each chain from the input distribution.

    Parameters
    ----------
    Zdict: dict
        A dictionary (as in MC3's output) containing a posterior distribution
        (Z) and number of iterations to burn (burnin).
    burnin: Integer
        Number of iterations to remove from the start of each chain.
        If specified, it overrides value from Zdict.
    Z: 2D float ndarray
        Posterior distribution (of shape [nsamples,npars]) to consider
        if Zdict is None.
    zchain: 1D integer ndarray
        Chain indices for the samples in Z (used only of Zdict is None).
    sort: Bool
        If True, sort the outputs by chain index.

    Returns
    -------
    posterior: 2D float ndarray
        Burned posterior distribution.
    zchain: 1D integer ndarray
        Burned zchain array.
    zmask: 1D integer ndarray
        Indices that transform Z into posterior.

    Examples
    --------
    >>> import mc3.utils as mu
    >>> import numpy as np
    >>> # Mock a posterior-distribution output:
    >>> Z = np.expand_dims([0., 1, 10, 20, 30, 11, 31, 21, 12, 22, 32], axis=1)
    >>> zchain = np.array([-1, -1, 0, 1, 2, 0, 2, 1, 0, 1, 2])
    >>> Zdict = {'posterior':Z, 'zchain':zchain, 'burnin':1}
    >>> # Simply apply burn() into the dict:
    >>> posterior, zchain, zmask = mu.burn(Zdict)
    >>> print(posterior[:,0])
    [11. 12. 21. 22. 31. 32.]
    >>> print(zchain)
    [0 0 1 1 2 2]
    >>> print(zmask)
    [ 5  8  7  9  6 10]
    >>> # Samples were sorted by chain index, but one can prevent with:
    >>> posterior, zchain, zmask = mu.burn(Zdict, sort=False)
    >>> print(posterior[:,0])
    [11. 31. 21. 12. 22. 32.]
    >>> # One can also override the burn-in samples:
    >>> posterior, zchain, zmask = mu.burn(Zdict, burnin=0)
    >>> print(posterior[:,0])
    [10. 11. 12. 20. 21. 22. 30. 31. 32.]
    >>> # Or apply directly to arrays:
    >>> posterior, zchain, zmask = mu.burn(Z=Z, zchain=zchain, burnin=1)
    >>> print(posterior[:,0])
    [11. 12. 21. 22. 31. 32.]
    """
    if Zdict is None and (Z is None or zchain is None or burnin is None):
        raise ValueError('Need to input either Zdict or all three of '
                         'burnin, Z, and zchain')
    if Zdict is not None:
        Z = Zdict['posterior']
        zchain = Zdict['zchain']

    if burnin is None:
        burnin = Zdict['burnin']

    mask = np.zeros_like(zchain, bool)
    nchains = np.amax(zchain) + 1
    for c in range(nchains):
        mask[np.where(zchain == c)[0][burnin:]] = True

    if sort:
        zsort = np.lexsort([zchain])
        zmask = zsort[np.where(mask[zsort])]
    else:
        zmask = np.where(mask)[0]

    # Values accepted for posterior stats:
    posterior = Z[zmask]
    zchain = zchain[zmask]

    return posterior, zchain, zmask


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
