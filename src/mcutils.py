# ******************************* START LICENSE *****************************
# 
# Multi-Core Markov-chain Monte Carlo (MC3), a code to estimate
# model-parameter best-fitting values and Bayesian posterior
# distributions.
# 
# This project was completed with the support of the NASA Planetary
# Atmospheres Program, grant NNX12AI69G, held by Principal Investigator
# Joseph Harrington.  Principal developers included graduate student
# Patricio E. Cubillos and programmer Madison Stemm.  Statistical advice
# came from Thomas J. Loredo and Nate B. Lust.
# 
# Copyright (C) 2014 University of Central Florida.  All rights reserved.
# 
# This is a test version only, and may not be redistributed to any third
# party.  Please refer such requests to us.  This program is distributed
# in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.
# 
# Our intent is to release this software under an open-source,
# reproducible-research license, once the code is mature and the first
# research paper describing the code has been accepted for publication
# in a peer-reviewed journal.  We are committed to development in the
# open, and have posted this code on github.com so that others can test
# it and give us feedback.  However, until its first publication and
# first stable release, we do not permit others to redistribute the code
# in either original or modified form, nor to publish work based in
# whole or in part on the output of this code.  By downloading, running,
# or modifying this code, you agree to these conditions.  We do
# encourage sharing any modifications with us and discussing them
# openly.
# 
# We welcome your feedback, but do not guarantee support.  Please send
# feedback or inquiries to:
# 
# Joseph Harrington <jh@physics.ucf.edu>
# Patricio Cubillos <pcubillos@fulbrightmail.org>
# 
# or alternatively,
# 
# Joseph Harrington and Patricio Cubillos
# UCF PSB 441
# 4111 Libra Drive
# Orlando, FL 32816-2385
# USA
# 
# Thank you for using MC3!
# ******************************* END LICENSE *******************************

import os, sys, time, traceback, textwrap, struct
import numpy as np
from numpy import array

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


def writedata(data, filename, rowwise=False):
  """
  Write data to file.

  Parameters:
  -----------
  data:  List or 1D/2D ndarray
     Data to be stored in file.
  filename:  String
     File where to store the arrlist.
  rowwise:  Boolean
     For an ndarray data type:
      - If True, store each row of data in a same line (empty-space separated)
      - If False, store each column of data in a same line.
     For a list data type:
      - If True: Store one value from each element of data in a same line.
      - If False, store each element of data in a same line.

  Notes:
  ------
  If rowwise is False, assume that every array in arrlist has the same
  number of elements.

  Examples:
  ---------
  >>> import numpy as np
  >>> import mcutils as mu

  >>> a = np.arange(7)*np.pi
  >>> b = True
  >>> c = -35e6
  >>> outfile = 'delete.me'
  >>> mu.writedata([a,b,c], outdata, True)

  >>> # This will produce this file:
  >>> f = open(outfile)
  >>> f.readlines()
  ['             0         3.14159         6.28319         9.42478         12.5664          15.708         18.8496\n',
   '             1\n',
   '       3.5e+07\n']
  >>> f.close()

  Modification History:
  ---------------------
  2014-05-03  patricio  Initial implementation.
  """
  # Force it to be a 2D ndarray:
  if not rowwise:
    if   type(data) in [list, tuple]:
      data = np.atleast_2d(np.array(data)).T
    elif type(data) == np.ndarray:
      data = np.atleast_2d(data).T

  # Force it to be a list of 1D arrays:
  else:
    if type(data) in [list, tuple]:
      for i in np.arange(len(data)):
        data[i] = np.atleast_1d(data[i])
    elif type(data) == np.ndarray and np.ndim(data) == 1:
      data = [data]

  # Save arrays to file:
  f = open(filename, "w")
  narrays = len(data)
  for i in np.arange(narrays):
    try:    # Fomat numbers
      f.write('  '.join('% 14.8g'% v for v in data[i]))
    except: # Non-numbers (Bools, ..., what else?)
      f.write('  '.join('% 14s'% str(v) for v in data[i]))
    f.write('\n')
  f.close()


def read2array(filename, square=True):
  """
  Extract data from file and store in a 2D ndarray (or list of arrays
  if not square).  Blank or comment lines are ignored.

  Parameters:
  -----------
  filename: String
     Path to file containing the data to be read.
  square: Boolean
     If True:  assume all lines contain the same number of (white-space
               separated) values, store the data in a transposed 2D ndarray.
     If False: Store the data in a list (one list-element per line), if
               there is more than one value per line, store as 1D ndarray.

  Returns:
  --------
  array: 2D ndarray or list
     See parameters description.

  Modification History:
  ---------------------
  2014-04-17  patricio  Initial implementation.
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
  if square:
    ncolumns = len(lines[0].split())
    array = np.zeros((nlines, ncolumns), np.double)
    for i in np.arange(nlines):
      array[i] = lines[i].strip().split()
    array = np.transpose(array)

  else:
    array = []
    for i in np.arange(nlines):
      values = lines[i].strip().split()
      if len(values) > 1:
        array.append(np.asarray(lines[i].strip().split(), np.double))
      else:
        array.append(np.double(values[0]))

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
      error("Object type not understood in file: '{:s}'".format(filename))
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


def comm_spawn(worker, nprocs, cfile, rargs=[], path=None):
  """
  Spawns

  Parameters:
  -----------
  worker: String
     Filename of the worker process to spawn.
  nprocs: Integer
     The number of processes to spawn.
  cfile: String
     Configuration file.
  rargs: List
     Remaining arguments.

  Modification History:
  ---------------------
  2014-03-24  Madison   Initial implementation. Madison Stemm, UCF.
  2014-04-13  patricio  Modified for BART project, documented.
                        pcubillos@fulbrightmail.org.
  2014-05-13  asdf      Modified to allow for direct spawning of c executables
                        andrew.scott.foster@gmail.com
  2014-12-13  patricio  Updated C calling command.
  """
  if path is not None:
    sys.path.append(path)
  if worker.endswith(".py"):
    args = [path + worker, "-c" + cfile] + rargs
    comm = MPI.COMM_SELF.Spawn(sys.executable, args=args, maxprocs=nprocs)
  else: # Assume that it's a binary executable:
    args = ["-c" + cfile] + rargs
    comm = MPI.COMM_SELF.Spawn(worker,         args=args, maxprocs=nprocs)
  return comm
  # FINDME: I'm thinking this function should not belong to MC3.

def comm_scatter(comm, array, mpitype=None):
  """
  Scatter to send or receive an MPI array.

  Parameters:
  -----------
  comm: MPI communicator
     The MPI Intracommunicator instance.
  array: 1D ndarray
     The array transferred.
  mpitype: MPI data type
     The data type of the array to be send (if not None). If None,
     assume it is receiving an array.  

  Notes:
  ------
  Determine wheter to send or receive an array depending on 'mpitype'

  Modification History:
  ---------------------
  2014-03-24  Madison   Initial implementation. Madison Stemm, UCF.
  2014-04-13  patricio  Documented.  pcubillos@fulbrightmail.org.
  2014-04-18  patricio  Joined master and worker routines.
  """
  comm.Barrier()
  if mpitype is None:  # Receive
    comm.Scatter(None, array, root=0)
  else:                # Send
    comm.Scatter([array, mpitype], None, root=MPI.ROOT)


def comm_gather(comm, array, mpitype=None):
  """
  Gather to send or receive an MPI array.

  Parameters:
  -----------
  comm: MPI communicatior
     The MPI Intracommunicator.
  array: 1D ndarray
     The array transferred.
  mpitype: MPI data type
     The data type of the array to be send (if not None). If None,
     assume it is receiving an array.

  Modification History:
  ---------------------
  2014-03-24  Madison   Initial implementation. Madison Stemm, UCF.
  2014-04-13  patricio  Documented.  pcubillos@fulbrightmail.org
  2014-04-18  patricio  Joined master and worker routines.
  """
  comm.Barrier()
  if mpitype is None:  # Receive
    comm.Gather(None, array,            root=MPI.ROOT)
  else:                # Send
    comm.Gather([array, mpitype], None, root=0)


def comm_bcast(comm, array, mpitype=None):
  """
  Broadcast to send or receive an MPI array.

  Parameters:
  -----------
  comm: MPI communicatior
     The MPI Intracommunicator.
  array: 1D ndarray
     The array transferred.
  mpitype: MPI data type
     The data type of the array to be send (if not None). If None,
     assume it is receiving an array.

  Modification History:
  ---------------------
  2014-04-18  patricio  Initial implementation. pcubillos@fulbrightmail.org
  """
  comm.Barrier()
  if mpitype is None:  # Receive
    comm.Bcast(array,            root=0)
  else:                # Send
    comm.Bcast([array, mpitype], root=MPI.ROOT)


def comm_disconnect(comm):
  """
  Close communication with comm.

  Parameters:
  -----------
  comm: MPI communicator
    An MPI Intracommmunicator.

  Modification History:
  ---------------------
  2014-05-02  patricio  Initial implementation.
  """
  if comm is not None:
    comm.Barrier()
    comm.Disconnect()


def msg(verblevel, message, file=None, indent=0, noprint=False):
  """
  Conditional message printing to screen.

  Modification History:
  ---------------------
  2014-06-15  patricio  Added Documentation.
  2014-08-18  patricio  Copied to BART project.
  2015-05-15  patricio  Added file and noprint arguments.
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
    # Print to file, if requested:
    if file is not None:
      file.write(text)


def warning(message, file=None):
  """
  Print message surrounded by colon bands.

  Modification History:
  ---------------------
  2014-06-15  patricio  Initial implementation.
  2014-08-18  patricio  Copied to BART project.
  2015-05-15  patricio  Added file argument.
  """
  text = ("{:s}\n  Warning:\n{:s}\n{:s}".
           format(sep, msg(1,message, indent=4,noprint=True)[:-1], sep))
  print(text)
  if file is not None:
    file.write(text + "\n")


def error(message, file=None):
  """
  Pretty print error message.

  Modification History:
  ---------------------
  2014-06-15  patricio  Initial implementation.
  2014-08-18  patricio  Copied to BART project.
  2015-05-15  patricio  Added file argument.
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
  # Print to file if requested:
  if file is not None:
    file.write(text)
    file.close()
  sys.exit(0)


def progressbar(frac, file=None):
   """
   Print out to screen a progress bar, percentage, and current time.

   Parameters:
   -----------
   frac: Float
      Fraction of the task that has been completed, ranging from 0.0 (none) 
      to 1.0 (completed).
   file: FILE pointer
      If not None, also print to the given file.

   Modification History:
   ---------------------
   2014-04-19  patricio  Initial implementation.
   2015-05-15  patricio  Added file argument.
   2015-05-15  patricio  Added file argument.
   """
   barlen = int(np.clip(10*frac, 0, 10))
   bar = ":"*barlen + " "*(10-barlen)
   text = "\n[%s] %5.1f%% completed  (%s)"%(bar, 100*frac, time.ctime())
   # Print to screen and to file:
   print(text)
   if file is not None:
     file.write(text + "\n")

