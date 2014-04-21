import sys, time
import numpy as np
from mpi4py import MPI


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
     If False: Store the data in a list (one element per line) of 1D ndarrays.

  Returns:
  --------
  array: 2D ndarray or list of 1D ndarray
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
      array.append(np.asarray(lines[i].strip().split(), np.double))

  return array


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
  """
  if message is not None:
    print(message)
  if comm is not None:
    if abort:
      comm_gather(comm, np.array([1]), MPI.INT)
    comm.Barrier()
    comm.Disconnect()
  if comm2 is not None:
    comm2.Barrier()
    comm2.Disconnect()
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

