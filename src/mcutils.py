# Copyright (c) 2015-2016 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

import os, sys, time, traceback, textwrap, struct
import numpy as np
from numpy import array

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
  >>> mu.saveascii([a,b,c], outfile)

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


def writebin(data, filename):
  """
  Write data to file in binary format, storing the objects type, data-type,
  and shape.

  Parameters:
  -----------
  data:  List of data objects
     Data to be stored in file.  Each array must have the same length.
  filename:  String
     File where to store the arrlist.

  Notes:
  ------
  - Known to work for multi-dimensional ndarrays, scalars, and booleans
    (at least).
  - Floating values are stored with double precision, integers are stored
    as long-integers.

  Examples:
  ---------
  >>> import numpy as np
  >>> import mcutils as mu

  >>> data = [np.arange(4),np.ones((2,2)), True, 42]
  >>> outfile = 'delete.me'
  >>> mu.writebin(data, outfile)

  Modification History:
  ---------------------
  2014-09-12  patricio  Initial implementation.
  """

  f = open(filename, "wb")
  # Number of data structures:
  ndata = len(data)
  # Object type:
  otype = np.zeros(ndata, np.int)
  # Data type:
  dtype = np.zeros(ndata, str)
  # Number of dimensions:
  ndim  = np.zeros(ndata, np.int)
  # List of data sizes:
  dsize = []

  # Storage data-type format:
  fmt = ["", "d", "l", "s", "?"]

  info  = struct.pack("h", ndata)
  # Read objects types and dimensions:
  for i in np.arange(ndata):
    # Determine the object type:
    otype[i] = (1*(type(data[i]) is float) + 2*(type(data[i]) is int  ) +
                3*(type(data[i]) is str  ) + 4*(type(data[i]) is bool ) +
                5*(type(data[i]) is list ) + 5*(type(data[i]) is tuple) +
                6*(type(data[i]) is np.ndarray) )
    # TBD: add NoneType
    if otype[i] == 0:
      exit(message="Object type not understood in file: '{:s}'".format(filename))
    info += struct.pack("h", otype[i])

    # Determine data dimensions:
    if   otype[i] < 5:
      ndim[i] = 1
    elif otype[i] == 5:
      ndim[i] = 1  # FIX-ME
    elif otype[i] == 6:
      ndim[i] = data[i].ndim
    info += struct.pack("h", ndim[i])

    # Determine the data type:
    if otype[i] < 5:
      dtype[i] = fmt[otype[i]]
    elif otype[i] ==6:
      dtype[i] = fmt[1*isinstance(data[i].flat[0], float) +
                     2*isinstance(data[i].flat[0], int)   +
                     3*isinstance(data[i].flat[0], str)   +
                     4*isinstance(data[i].flat[0], bool)  ]
    info += struct.pack("c", dtype[i])

    # Determine the dimension lengths (shape):
    if otype[i] < 5:
      dsize.append([1,])
    elif otype[i] == 5:
      shape = []
      for j in np.arange(ndim[i]):
        shape.append(len(data[i]))
      dsize.append(shape)
    elif otype[i] == 6:
      dsize.append(np.shape(data[i]))
    info += struct.pack("{}i".format(ndim[i]), *dsize[i])

  # Write the data:
  for i in np.arange(ndata):
    if   otype[i] < 5:
      info += struct.pack(dtype[i], data[i])
    elif otype[i] == 5:
      info += struct.pack("{}{}".format(len(data), dtype[i]), *data[i])
    elif otype[i] == 6:
      info += struct.pack("{}{}".format(data[i].size, dtype[i]),
                                        *list(data[i].flat))

  f.write(info)
  f.close()


def readbin(filename):
  """
  Read a binary file and extract the data objects.

  Parameters:
  -----------
  filename: String
     Path to file containing the data to be read.

  Return:
  -------
  data:  List
     List of objects stored in the file.

  Example:
  --------
  >>> import mutils as mu
  >>> # Continue example from writebin():
  >>> v = mu.read2list("delete.me")
      [array([0, 1, 2, 3]), array([[ 1.,  1.], [ 1.,  1.]]), True, 42]

  Modification History:
  ---------------------
  2014-09-12  patricio  Initial implementation.
  """
  f = open(filename, "rb")

  # Read number of data objects:
  ndata  = struct.unpack("h", f.read(2))[0]

  # Object type:
  otype = np.zeros(ndata, np.int)
  # Data type:
  dtype = np.zeros(ndata, str)
  # Number of dimensions:
  ndim  = np.zeros(ndata, np.int)
  # List of data sizes:
  dsize = []

  for i in np.arange(ndata):
    # Read the object type:
    otype[i] = struct.unpack("h", f.read(2))[0]
    # Read the data dimensions:
    ndim[i]  = struct.unpack("h", f.read(2))[0]
    # Read the data type:
    dtype[i] = struct.unpack("c", f.read(1))[0]
    # Read the shape:
    dsize.append(struct.unpack("{}i".format(ndim[i]), f.read(4*ndim[i])))

  # Read data:
  data = []
  for i in np.arange(ndata):
    fmt  = "{}{}".format(np.prod(dsize[i]), dtype[i])
    size = struct.calcsize(dtype[i]) * np.prod(dsize[i])
    d = struct.unpack(fmt, f.read(size))
    if   otype[i] <  5:
      data.append(d[0])
    elif otype[i] == 6:
      data.append(np.reshape(d, dsize[i]))
  f.close()

  return data


def msg(verblevel, message, file=None, indent=0, noprint=False):
  """
  Conditional message printing to screen and to file.

  Parameters:
  -----------
  verblevel: Integer
     Conditional threshold to print the message.  Print only if
     verblevel is positive.
  message: String
     String to be printed.
  file: File pointer
     If not None, print message to the given file pointer.
  indent: Integer
     Number of blank spaces to indent the printed message.
  noprint: Boolean
     If True, do not print and return the string instead.

  Returns:
  --------
  text: String
     If noprint is True, return the formatted output string.
  """
  if verblevel <= 0:
    return

  # Output text to be printed:
  text = ""
  # Break down the input text into the different sentences (line-breaks):
  sentences = message.splitlines()
  # Make the indentation blank spaces:
  indspace = " "*indent

  for s in sentences:
    msg = textwrap.fill(s, break_long_words=False, initial_indent=indspace,
                                                subsequent_indent=indspace)
    text += msg + "\n"

  # Do not print, just return the string:
  if noprint:
    return text
  else:
    # Print to screen:
    print(text[:-1])  # Remove the trailing line-break
    sys.stdout.flush()
    if file is not None:
      file.write(text)


def warning(message, file=None):
  """
  Print message surrounded by colon bands.

  Parameters:
  -----------
  message: String
     String to be printed.
  file: File pointer
     If not None, print message to the given file pointer.
  """
  # Format the sub-text message:
  subtext = msg(1, message, indent=4, noprint=True)[:-1]
  # Add the warning surroundings:
  text = "{:s}\n  Warning:\n{:s}\n{:s}".format(sep, subtext, sep)

  # Print to screen:
  print(text)
  sys.stdout.flush()
  if file is not None:  # And print to file:
    file.write(text + "\n")


def error(message, file=None):
  """
  Pretty-print error message and end the code execution.

  Parameters:
  -----------
  message: String
     String to be printed.
  file: File pointer
     If not None, print message to the given file pointer.
  """
  # Trace back the file, function, and line where the error source:
  trace = traceback.extract_stack()
  # Extract fields:
  modpath  = trace[-2][0]
  modname  = modpath[modpath.rfind('/')+1:]
  funcname = trace[-2][2]
  linenum  = trace[-2][1]

  # Generate string to print:
  subtext = msg(1, message, indent=4, noprint=True)[:-1]
  text = ("{:s}\n  Error in module: '{:s}', function: '{:s}', line: {:d}\n"
          "{:s}\n{:s}".format(sep, modname, funcname, linenum, subtext, sep))

  # Print to screen:
  print(text)
  sys.stdout.flush()
  # Print to file and close, if exists:
  if file is not None:
    file.write(text)
    file.close()
  sys.exit(0)


def progressbar(frac, file=None):
  """
  Print out to screen [and file] a progress bar, percentage,
  and current time.

  Parameters:
  -----------
  frac: Float
     Fraction of the task that has been completed, ranging from 0.0 (none)
     to 1.0 (completed).
  file: File pointer
     If not None, print message to the given file pointer.
  """
  barlen = int(np.clip(10*frac, 0, 10))
  bar = ":"*barlen + " "*(10-barlen)

  text = "\n[%s] %5.1f%% completed  (%s)"%(bar, 100*frac, time.ctime())
  # Print to screen and to file:
  print(text)
  sys.stdout.flush()
  if file is not None:
    file.write(text + "\n")
