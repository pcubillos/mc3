import os, sys, time, traceback, textwrap
import numpy as np
from numpy import array

try:
  from mpi4py import MPI
except:
  pass

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


def writerepr(data, filename):
  """
  Write string representation of elements is data to a file.

  Parameters:
  -----------
  data: List
     List of data to be stored.
  file: String
     File where to store the arrlist.

  Notes:
  ------
  Known supported structures:
    - Numpy arrays of any dimension
    - Booleans
    - Strings
    - list, list of lists, etc.

  Example:
  --------
  >>> import mutils as mu

  >>> # Make variables to store:
  >>> a = np.arange(10)
  >>> b = [[1], [2,3], []]
  >>> c = True
  >>> d = np.array([[0,1],[2,3]])
  >>> e = "Hello World"
  >>> f = 'Hola Mundo'

  >>> # Write to file:
  >>> mu.writerepr([a,b,c,d, e, f], "vars.dat")

  Modification History:
  ---------------------
  2014-09-02  patricio  Initial implementation
  """
  # Set numpy option to print all elements in the array:
  np.set_printoptions(threshold=np.inf)

  # Open file to write:
  f = open(filename, "w")
  f.write("#repr\n")
  # For each element in data write one line with the element's string
  # representation (i.e., remove line breaks from string):
  for element in data:
    f.write(repr(element).replace("\n",  "") + "\n")
  f.close()


def read2list(filename):
  """
  Read a file and extract data stored as string representation, return each
  line from the file as an item in a list.

  Parameters:
  -----------
  filename: String
     Path to file containing the data to be read.

  Return:
  -------
  data:  List
     List of variables stored in the file.

  Example:
  --------
  >>> import mutils as mu
  >>> # Continue example from writerepr():
  >>> v = mu.read2list("vars.dat")

  Modification History:
  ---------------------
  2014-09-02  patricio  Initial implementation.
  """
  f = open(filename, "r")

  data = []
  line = f.readline()
  while line != "":
    # Get data, skip comments and empty lines:
    if not line.startswith('#') and line.strip() != "":
      exec("data.append({:s})".format(line))
    line = f.readline()

  f.close()
  return data


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


def exit(comm=None, abort=False, message=None, comm2=None):
  """
  Stop execution.

  Parameters:
  -----------
  comm: MPI communicator
     An MPI Intracommunicator.
  abort: Boolean
     If True send (gather) an abort flag integer through comm.
  message: String
     Print message on exit.

  Modification History:
  ---------------------
  2014-04-20  patricio  Initial implementation. pcubillos@fulbrightmail.org
  2014-05-04  patricio  Improved message printing with traceback and textwrap.
  """
  if message is not None:
    # Trace back the file, function, and line where the error source:
    t = traceback.extract_stack()
    # Extract fields:
    modpath = t[-2][0]  # Module path
    modulename = modpath[modpath.rfind('/')+1:] # Module name
    funcname   = t[-2][2] # Function name
    linenumber = t[-2][1] # Line number
    # Indent and wrap message to 70 characters:
    msg = textwrap.fill(message, initial_indent   ="    ",
                                 subsequent_indent="    ")
    print("\n"
    "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n"
    "  Error in module: '%s', function: '%s', line: %d\n"
    "%s\n"
    "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n"%
    (modulename, funcname, linenumber, msg))
  if comm is not None:
    if abort:
      comm_gather(comm, np.array([1]), MPI.INT)
    comm_disconnect(comm)
  if comm2 is not None:
    comm_disconnect(comm2)
  sys.exit(0)


def progressbar(frac):
   """
   Print out to screen a progress bar, percentage and current time.

   Parameters:
   -----------
   frac: Float
      Fraction of the task that has been completed, ranging from 0.0 (none) 
      to 1.0 (completed).

   Modification History:
   ---------------------
   2014-04-19  patricio  Initial implementation.
   """
   barlen = int(np.clip(10*frac, 0, 10))
   bar = ":"*barlen + " "*(10-barlen)
   print("\n[%s] %5.1f%% completed  (%s)"%(bar, 100*frac, time.ctime()))

