import sys
import os

import multiprocessing as mp
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/cfuncs/lib')
import chisq    as cs

class Chain(mp.Process):
  """
  Background process.  This guy evaluates the model, and calculates chisq.
  """
  def __init__(self, func, args, requests, results, data, uncert,
               timeout=None, **kwds):
    """
    Parameters:
    -----------
    func: Callable
    requests: Queue.Queue instance
    results: Queue.Queue instance
    timeout: Float
    """
    mp.Process.__init__(self, **kwds)
    self.daemon   = True
    self.func     = func
    self.args     = args
    self.requests = requests
    self.results  = results
    self.data     = data
    self.uncert   = uncert
    self.timeout  = timeout
    self.start()
    print("Initialized chain")


  def run(self):
    """
    Process the requests queue until told to exit.
    """
    print("Started chain!")
    while True:

      # Process next request from the queue:
      try:
        print("Waiting ... ")
        req = self.requests.get(True, self.timeout)
        print("New request!:  {}".format(req))
        # Stop the loop if the dismiss flag is True:
        if req == -1:
          break
      except q.Empty:
        continue
      else:
        try:
          # Evaluate callable:
          print("Evaluate model.")
          result = self.func(req, *self.args)
          # Calculate chisq:
          print("Calculate chisq.")
          chisq = cs.chisq(result, self.data, self.uncert)
          print("chisq = {}".format(chisq))
          # Put chisq in the results Queue:
          self.results.put(chisq)
          print("Put in queue.")
        except:
          self.results.put(-1)


class Mcpool:
  """
  A process pool.
  """
  def __init__(self, nchains):
    """
    """
    # Initialize the object's variables:
    self.nchains = nchains
    self.request = mp.Queue()
    self.results = mp.Queue()
    self.chains = []
    # Initialize the chain workers:
    self.startChains()


  def startChains(self, timeout=None):
    """
    Docstring.
    """
    for i in np.arange(self.nchains):
      self.chains.append(Chain(self.func, self.args,
                               self.requests, self.results,
                               self.data, self.uncert,
                               timeout=timeout))


