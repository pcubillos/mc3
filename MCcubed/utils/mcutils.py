# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ["parray", "saveascii", "loadascii", "savebin", "loadbin",
           "isfile", "binarray", "weightedbin", "credregion",
           "default_parnames"]

import os
import sys
import numpy as np
import scipy.stats as stats
import scipy.interpolate as si

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../lib')
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

  >>> a = np.arange(1,5)*np.pi
  >>> b = np.ones(4)
  >>> c = [10, 5, -5, -9.9]
  >>> outfile = 'delete.me'
  >>> mu.saveascii([a,b,c], outfile)

  >>> # This will produce this file:
  >>> with open(outfile) as f:
  >>>   print(f.read())
  3.1415927         1        10
  6.2831853         1         5
   9.424778         1        -5
  12.566371         1      -9.9
  """
  # Force it to be a 2D ndarray:
  data = np.array(data, ndmin=2).T

  # Save arrays to ASCII file:
  with open(filename, "w") as f:
    for parvals in data:
      f.write(' '.join("{:9.8g}".format(v) for v in parvals) + '\n')


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
  with open(filename, "r") as f:
    for line in f:
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
  >>> datafile = "datafile.npz"
  >>> indata = [np.arange(4), "one", np.ones((2,2)), True, [42], (42, 42)]
  >>> mu.savebin(indata, datafile)
  >>> # Now load the file:
  >>> outdata = mu.loadbin(datafile)
  >>> for data in outdata:
  >>>   print(repr(data))
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
    dkey = "file{:{}d}".format(i, fmt)
    # Encode in the key if a variable is a list or tuple:
    if isinstance(datum, list):
      dkey += "_list"
    if isinstance(datum, tuple):
      dkey += "_tuple"
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
    # Check if val is a list or tuple:
    if key.count("_"):
      exec("data[-1] = " + key[key.find('_')+1:] + "(data[-1])")

  return data


def isfile(input, iname, log, dtype, unpack=True, notnone=False):
  """
  Check if an input is a file name; if it is, read it.
  Genereate error messages if it is the case.

  Parameters
  ----------
  input: Iterable or String
    The input variable.
  iname:  String
    Input-variable  name.
  log: File pointer
     If not None, print message to the given file pointer.
  dtype:  String
    File data type, choose between 'bin' or 'ascii'.
  unpack:  Bool
    If True, return the first element of a read file.
  notnone:  Bool
    If True, throw an error if input is None.
  """
  # Set the loading function depending on the data type:
  if   dtype == "bin":
    load = loadbin
  elif dtype == "ascii":
    load = loadascii
  else:
    log.error("Invalid data type '{:s}', must be either 'bin' or 'ascii'.".
              format(dtype), lev=-3)

  # Check if the input is None, throw error if requested:
  if input is None:
    if notnone:
      log.error("'{:s}' is a required argument.".format(iname), lev=-3)
    return None

  # Check that it is an iterable:
  if not np.iterable(input):
    log.error("{:s} must be an iterable or a file name.".format(iname), lev=-3)

  # Check if it is a string:
  if isinstance(input, str):
    ifile = input

  # Check if first element is a string:
  elif isinstance(input[0], str):
    ifile = input[0]

  # It is an array of values:
  else:
    return input

  # It is a file name:
  if not os.path.isfile(ifile):
    log.error("{:s} file '{:s}' not found.".format(iname, ifile), lev=-3)
  else:
    if unpack:  # Unpack (remove outer dimension) if necessary
      return load(ifile)[0]
    return load(ifile)


def credregion(posterior= None,        percentile= [0.6827, 0.9545], 
               pdf      = None,        xpdf      = None, 
               lims     = (None,None), numpts    = 200):
  """
  Compute a smoothed posterior density distribution and the minimum
  density for a given percentile of the highest posterior density.

  These outputs can be used to easily compute the HPD credible regions.

  Parameters
  ----------
  posterior: 1D float ndarray
     A posterior distribution.
  percentile: 1D float ndarray, list, or float.
     The percentile (actually the fraction) of the credible region.
     A value in the range: (0, 1).
  pdf: 1D float ndarray
     A smoothed-interpolated PDF of the posterior distribution.
  xpdf: 1D float ndarray
     The X location of the pdf values.
  lims: tuple, floats
     Minimum and maximum allowed values for posterior. Should only be used if 
     there are physically-imposed limits.
  numpts: int.
     Number of points to use when calculating the PDF.

  Returns
  -------
  pdf: 1D float ndarray
     A smoothed-interpolated PDF of the posterior distribution.
  xpdf: 1D float ndarray
     The X location of the pdf values.
  regions: list of 2D float ndarrays
     The values of the credible regions specified by `percentile`.
     regions[0] corresponds to percentile[0], etc.
     regions[0][0] gives the start and stop values of the first region of the CR
     regions[0][1] gives the second CR start/stop values, if the CR is composed 
                   of disconnected regions

  Example
  -------
  >>> import numpy as np
  >>> npoints = 100000
  >>> posterior = np.random.normal(0, 1.0, npoints)
  >>> pdf, xpdf, HPDmin = credregion(posterior)
  >>> # 68% HPD credible-region boundaries (somewhere close to +/-1.0):
  >>> print(np.amin(xpdf[pdf>HPDmin]), np.amax(xpdf[pdf>HPDmin]))

  >>> # Re-compute HPD for the 95% (withour recomputing the PDF):
  >>> pdf, xpdf, HPDmin = credregion(pdf=pdf, xpdf=xpdf, percentile=0.9545)
  >>> print(np.amin(xpdf[pdf>HPDmin]), np.amax(xpdf[pdf>HPDmin]))
  """
  # Make sure `percentile` is a list or array
  if type(percentile) == float:
    percentile = np.array([percentile])
  if pdf is None and xpdf is None:
    # Compute the posterior's PDF:
    kernel = stats.gaussian_kde(posterior)
    # Use a Gaussian kernel density estimate to trace the PDF:
    # Interpolate-resample over finer grid (because kernel.evaluate
    #  is expensive):
    lo   = np.amin(posterior)
    hi   = np.amax(posterior)
    x    = np.linspace(lo, hi, numpts)
    f    = si.interp1d(x, kernel.evaluate(x))
    xpdf = np.linspace(lo, hi, 100*numpts)
    pdf  = f(xpdf)

  # Sort the PDF in descending order:
  ip = np.argsort(pdf)[::-1]
  # Sorted CDF:
  cdf = np.cumsum(pdf[ip])

  # List to hold boundaries of CRs
  # List is used because a given CR may be multiple disconnected regions
  regions = []
  # Find boundary for each specified percentile
  for i in range(len(percentile)):
    # Indices of the highest posterior density:
    iHPD = np.where(cdf >= percentile[i]*cdf[-1])[0][0]
    # Minimum density in the HPD region:
    HPDmin   = np.amin(pdf[ip][0:iHPD])
    # Find the contiguous areas of the PDF greater than or equal to HPDmin
    HPDbool  = pdf >= HPDmin
    idiff    = np.diff(HPDbool) # True where HPDbool changes T to F or F to T
    iregion, = idiff.nonzero()  # Indexes of Trues. Note , because returns tuple
    # Check boundaries
    if HPDbool[0]:
      iregion = np.insert(iregion, 0, -1) # This -1 is changed to 0 below when 
    if HPDbool[-1]:                       #   correcting start index for regions
      iregion = np.append(iregion, len(HPDbool)-1)
    # Reshape into 2 columns of start/end indices
    iregion.shape = (-1, 2)
    # Add 1 to start of each region due to np.diff() functionality
    iregion[:,0] += 1
    regions.append(xpdf[iregion])

  return pdf, xpdf, regions


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
  return np.array(["Param {:0{}d}".format(i+1,namelen) for i in range(npars)])
