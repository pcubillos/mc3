import sys
import os

import multiprocessing as mp
import Queue as Queue
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/cfuncs/lib')
import chisq as cs
import dwt   as dwt

class Chain(mp.Process):
  """
  Background process.  This guy evaluates the model, and calculates chisq.
  """
  def __init__(self, func, args, pipe, data, uncert,
               wlike, prior, priorlow, priorup, csscale,
               timeout=None, **kwds):
    """
    Doc me!

    Parameters:
    -----------
    func: Callable
    requests: Queue.Queue instance
    results: Queue.Queue instance
    timeout: Float
    """
    # Multiprocessing setup:
    mp.Process.__init__(self, **kwds)
    self.daemon   = True
    self.timeout  = timeout
    # Modeling function:
    self.func     = func
    self.args     = args
    # Input/output Pipe:
    self.pipe     = pipe
    # Data:
    self.data     = data
    self.uncert   = uncert
    # Chisq function:
    self.wlike    = wlike
    self.csscale  = csscale
    # Priors:
    self.iprior   = np.where(priorlow != 0)
    self.prior    = prior   [self.iprior]  # Keep only the ones that count
    self.priorlow = priorlow[self.iprior]
    self.priorup  = priorup [self.iprior]
    # Start the Process:
    self.start()


  def run(self):
    """
    Process the requests queue until told to exit.
    """
    #print("Started chain!")
    while True:
      # Process next request from the queue:
      try:
        #print("Waiting ... ")
        req = self.pipe.recv()
        ID = 0
        # Stop the loop if the dismiss flag is True:
        if ID == -1:
          #print("Terminate '{}'.".format(self.name))
          break
        # Take chisq normalizing factor:
        if ID == -2:
          self.uncert *= req
          sleep(0.5)
          # FINDME: Print statement with ID to make sure it works
          continue

      except Queue.Empty:
        continue

      else:
        try:
          # Evaluate callable:
          if self.wlike:
            model = self.func(req[0:-3], *self.args)
          else:
            model = self.func(req, *self.args)
          # Calculate prioroff = params-prior:
          prioroff = req[self.iprior] - self.prior
          # Calculate chisq:
          if self.wlike:
            chisq = dwt.wlikelihood(req[-3:], model, self.data, prioroff,
                                    self.priorlow, self.priorup)
          else:
            chisq = cs.chisq(model, self.data, self.uncert, prioroff,
                             self.priorlow, self.priorup)
          #print("chisq = {}".format(chisq))
          # Put chisq in the results Queue:
          self.pipe.send(chisq)
        except:
          self.pipe.send(-1)

