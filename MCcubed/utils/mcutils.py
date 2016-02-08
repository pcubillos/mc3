# Copyright (c) 2015-2016 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ["sep", "parray", "saveascii", "loadascii", "savebin", "loadbin",
           "msg", "warning", "error", "progressbar", "isfile"]

import os, sys
import time
import traceback
import textwrap
import struct
import numpy as np

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


def msg(verblevel, message, file=None, indent=0, noprint=False):
  """
  Conditional message printing to screen and to file.

  Parameters
  ----------
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

  Returns
  -------
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

  Parameters
  ----------
  message: String
     String to be printed.
  file: File pointer
     If not None, print message to the given file pointer.
  """
  # Format the sub-text message:
  subtext = msg(1, message, indent=4, noprint=True)[:-1]
  # Add the warning surroundings:
  text = "\n{:s}\n  Warning:\n{:s}\n{:s}\n".format(sep, subtext, sep)

  # Print to screen:
  print(text)
  sys.stdout.flush()
  if file is not None:  # And print to file:
    file.write(text + "\n")


def error(message, file=None):
  """
  Pretty-print error message and end the code execution.

  Parameters
  ----------
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

  Parameters
  ----------
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


def isfile(input, iname, file, dtype, unpack=True, notnone=False):
  """
  Check if an input is a file name; if it is, read it.
  Genereate error messages if it is the case.

  Parameters
  ----------
  input: Iterable or String
    The input variable.
  iname:  String
    Input-variable  name.
  file: File pointer
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
    error("Invalid data type '{:s}', must be either 'bin' or 'ascii'.".
          format(dtype), file)

  # Check if the input is None, throw error if requested:
  if input is None:
    if notnone:
      error("'{:s}' is a required argument.".format(iname), log)
    return None

  # Check that it is an iterable:
  if not np.iterable(input):
    error("{:s} must be an iterable or a file name.".format(iname), log)

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
    error("{:s} file '{:s}' not found.".format(iname, ifile), log)
  else:
    if unpack:  # Unpack (remove outer dimension) if necessary
      return load(ifile)[0]
    return load(ifile)
