# Copyright (c) 2015-2018 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ["parray", "saveascii", "loadascii", "savebin", "loadbin",
           "comm_scatter", "comm_gather", "comm_bcast", "comm_disconnect",
           "msg", "warning", "error", "progressbar", "sep", "cred2ess"]

import os, sys, time, traceback, textwrap, struct
import numpy as np
import scipy.stats as ss

try:
  from mpi4py import MPI
except:
  pass

# Warning separator:
sep = 70*":"


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
  >>> import mcutils as mu

  >>> a = np.arange(1,5)*np.pi
  >>> b = np.ones(4)
  >>> c = [10, 5, -5, -9.9]
  >>> outfile = 'delete.me'
  >>> mu.writedata([a,b,c], outfile)

  >>> # This will produce this file:
  >>> f = open(outfile)
  >>> print(f.read())
  3.1415927         1        10
  6.2831853         1         5
   9.424778         1        -5
  12.566371         1      -9.9
  >>> f.close()
  """

  # Force it to be a 2D ndarray:
  data = np.array(data, ndmin=2).T

  # Save arrays to ASCII file:
  f = open(filename, "w")
  narrays = len(data)
  for i in np.arange(narrays):
    f.write(' '.join("{:9.8g}".format(v) for v in data[i]))
    f.write('\n')
  f.close()


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
  f = open(filename, "r")
  lines = f.readlines()
  f.close()

  # Remove comments and empty lines:
  nlines = len(lines)
  for i in np.arange(nlines, 0, -1):
    line = lines[i-1].strip()
    if line.startswith('#') or line == '':
      dummy = lines.pop(i-1)

  # Re-count number of lines:
  nlines = len(lines)

  # Extract values:
  ncolumns = len(lines[0].split())
  array = np.zeros((nlines, ncolumns), np.double)
  for i in np.arange(nlines):
    array[i] = lines[i].strip().split()
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
  >>> import mcutils as mu
  >>> import numpy as np

  >>> # Save list of data variables to file:
  >>> datafile = "datafile.npz"
  >>> indata = [np.arange(4), "one", np.ones((2,2)), True, [42], (42, 42)]
  >>> mu.savebin(indata, datafile)

  >>> # Now load the file:
  >>> outdata = mu.loadbin(datafile)
  >>> for i in np.arange(len(outdata)):
  >>>   print(repr(outdata[i]))
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
  for i in np.arange(ndata):
    dkey = "file{:{}d}".format(i, fmt)
    # Encode in the key if a variable is a list or tuple:
    if isinstance(data[i], list):
      dkey += "_list"
    if isinstance(data[i], tuple):
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


def comm_scatter(comm, array, mpitype=None):
  """
  Scatter to send or receive an MPI array.

  Parameters
  ----------
  comm: MPI communicator
     The MPI Intracommunicator instance.
  array: 1D ndarray
     The array transferred.
  mpitype: MPI data type
     The data type of the array to be send (if not None). If None,
     assume it is receiving an array.

  Notes
  -----
  Determine wheter to send or receive an array depending on 'mpitype'

  Uncredited developers
  ---------------------
  Madison Stemm  (UCF)
  """
  comm.Barrier()
  if mpitype is None:  # Receive
    comm.Scatter(None, array, root=0)
  else:                # Send
    comm.Scatter([array, mpitype], None, root=MPI.ROOT)


def comm_gather(comm, array, mpitype=None):
  """
  Gather to send or receive an MPI array.

  Parameters
  ----------
  comm: MPI communicatior
     The MPI Intracommunicator.
  array: 1D ndarray
     The array transferred.
  mpitype: MPI data type
     The data type of the array to be send (if not None). If None,
     assume it is receiving an array.

  Uncredited developers
  ---------------------
  Madison Stemm  (UCF)
  """
  comm.Barrier()
  if mpitype is None:  # Receive
    comm.Gather(None, array,            root=MPI.ROOT)
  else:                # Send
    comm.Gather([array, mpitype], None, root=0)


def comm_bcast(comm, array, mpitype=None):
  """
  Broadcast to send or receive an MPI array.

  Parameters
  ----------
  comm: MPI communicatior
     The MPI Intracommunicator.
  array: 1D ndarray
     The array transferred.
  mpitype: MPI data type
     The data type of the array to be send (if not None). If None,
     assume it is receiving an array.
  """
  comm.Barrier()
  if mpitype is None:  # Receive
    comm.Bcast(array,            root=0)
  else:                # Send
    comm.Bcast([array, mpitype], root=MPI.ROOT)


def comm_disconnect(comm):
  """
  Close communication with comm.

  Parameters
  ----------
  comm: MPI communicator
    An MPI Intracommmunicator.
  """
  if comm is not None:
    comm.Barrier()
    comm.Disconnect()


def msg(verblevel, message, file=None, indent=0, noprint=False):
  """
  Conditional message printing to screen.

  Parameters
  ----------
  verblevel: Integer
     If positive, print the given message.
  message: String
     Message to print.
  file: File pointer
     If not None, print message to the given file pointer.
  indent: Integer
     Number of blank spaces for indentation.
  noprint: Boolean
     If True, do not print and return the string instead.
  """
  if verblevel <= 0:
    return

  sentences = message.splitlines()
  indspace = " "*indent
  text = ""
  # Break the text down into the different sentences (line-breaks):
  for s in sentences:
    msg = textwrap.fill(s, break_long_words=False, initial_indent=indspace,
                                                subsequent_indent=indspace)
    text += msg + "\n"

  # Do not print, just return the string:
  if noprint:
    return text

  else:
    # Print to screen:
    print(text[:-1])  # Remove the trailing "\n"
    sys.stdout.flush()
    # Print to file, if requested:
    if file is not None:
      file.write(text)


def warning(message, file=None):
  """
  Print message surrounded by colon bands.

  Parameters
  ----------
  message: String
     Message to print.
  file: File pointer
     If not None, also print to the given file.
  """
  text = ("\n{:s}\n  Warning:\n{:s}\n{:s}".
           format(sep, msg(1,message, indent=4,noprint=True)[:-1], sep))
  print(text)
  sys.stdout.flush()
  if file is not None:
    file.write(text + "\n")


def error(message, file=None):
  """
  Pretty print error message.

  Parameters
  ----------
  message: String
     Message to print.
  file: File pointer
     If not None, also print to the given file.
  """
  # Trace back the file, function, and line where the error source:
  t = traceback.extract_stack()
  # Extract fields:
  modpath    = t[-2][0]                        # Module path
  modname    = modpath[modpath.rfind('/')+1:]  # Module name
  funcname   = t[-2][2]                        # Function name
  linenumber = t[-2][1]                        # Line number

  # Text to print:
  text = ("{:s}\n  Error in module: '{:s}', function: '{:s}', line: {:d}\n"
          "{:s}\n{:s}".format(sep, modname, funcname, linenumber,
                              msg(1,message,indent=4,noprint=True)[:-1], sep))

  # Print to screen:
  print(text)
  sys.stdout.flush()
  # Print to file if requested:
  if file is not None:
    file.write(text)
    file.close()
  sys.exit(0)


def progressbar(frac, file=None):
  """
  Print out to screen a progress bar, percentage, and current time.

  Parameters
  ----------
  frac: Float
     Fraction of the task that has been completed, ranging from 0.0 (none)
     to 1.0 (completed).
  file: File pointer
     If not None, also print to the given file.
  """
  barlen = int(np.clip(10*frac, 0, 10))
  bar = ":"*barlen + " "*(10-barlen)
  text = "\n[%s] %5.1f%% completed  (%s)"%(bar, 100*frac, time.ctime())
  # Print to screen and to file:
  print(text)
  sys.stdout.flush()
  if file is not None:
    file.write(text + "\n")

def cred2ess(p, eps):
  """
  Compute the Effective Sample Size (ESS) needed to compute
  a credible region with size p (in range 0 to 1), targeting
  a relative accuracy in 1-p of eps.

  Parameters
  ----------
  p:   Float
    Credible region size, from 0 to 1. E.g., 0.95 would indicate
    a 95% credible region.
  eps: Float
    Desired relative accuracy in 1-p. E.g., for p = 0.95, an
    eps of 0.02 would equate to 0.02 * 0.05 = 0.1% accuracy.

  Returns
  -------
  ess: Float
    ESS needed to meet eps accuracy on 1-p.
  """
  ess = 2*(ss.norm.ppf(0.5 * (1 - p)) / eps)**2
  return ess

