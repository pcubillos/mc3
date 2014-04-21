import numpy as np
import matplotlib.pyplot as plt

def trace(allparams, title=None, parname=None, thinning=1,
         fignum=-10, savefile=None, fmt="."):
  """
  Plot parameter trace MCMC sampling

  Parameters:
  -----------
  allparams: 2D ndarray
     An MCMC sampling array with dimension (number of parameters,
     sampling length).
  title: String
     Plot title.
  parname: Iterable (strings)
     List of label names for parameters.  If None use ['P0', 'P1', ...].
  thinning: Integer
     Thinning factor for plotting (plot every thinning-th value).
  fignum: Integer
     The figure number.
  savefile: Boolean
     If not None, name of file to save the plot.
  fmt: String
     The format string for the line and marker.

  Modification History:
  ---------------------
  2007-2012?  kevin     Initial version by Kevin Stevenson, UCF.
  2014-04-19  patricio  Updated and documented. pcubillos@fulbrightmail.org
  """
  # Get number of parameters and length of chain:
  npars, niter = np.shape(allparams)
  fs = 14

  # Set default parameter names:
  if parname is None:
    namelen = int(2+np.log10(npars-1))
    parname = np.zeros(npars, "|S%d"%namelen)
    for i in np.arange(npars):
      parname[i] = "P" + str(i).zfill(namelen-1)

  # Make the trace plot:
  plt.figure(fignum, figsize=(8,8))
  plt.clf()
  if title is not None:
    plt.suptitle(title, size=16)

  plt.subplots_adjust(left=0.15, right=0.95, bottom=0.10, top=0.90,
                      hspace=0.15)

  for i in np.arange(npars):
    a = plt.subplot(npars, 1, i+1)
    plt.plot(allparams[i, 0::thinning], fmt)
    plt.ylabel(parname[i], size=fs, multialignment='center')
    plt.yticks(size=fs)
    if i == npars - 1:
      plt.xticks(size=fs)
      if thinning > 1:
        plt.xlabel('MCMC (thinned) iteration', size=fs)
      else:
        plt.xlabel('MCMC iteration', size=fs)
    else:
      plt.xticks(visible=False)

  if savefile != None:
      plt.savefig(savefile)

def pairwise(allparams, title=None, parname=None, thinning=1,
             fignum=-20, savefile=None, style="hist"):
  """
  Plot parameter pairwise posterior distributions

  Parameters:
  -----------
  allparams: 2D ndarray
     An MCMC sampling array with dimension (number of parameters, 
     sampling length).
  title: String
     Plot title.
  parname: Iterable (strings)
     List of label names for parameters.  If None use ['P0', 'P1', ...].
  thinning: Integer
     Thinning factor for plotting (plot every thinning-th value).
  fignum: Integer
     The figure number.
  savefile: Boolean
     If not None, name of file to save the plot.
  style: String
     Choose between 'hist' to plot as histogram, or 'points' to plot
     the individual points.

  Modification History:
  ---------------------
  2007-2012?  kevin     Initial version by Kevin Stevenson, UCF.
  2013-??-??  ryan      Updated by Ryan Hardy.
  2014-04-19  patricio  Re-updated and documented. pcubillos@fulbrightmail.org
  """
  # Get number of parameters and length of chain:
  npars, niter = np.shape(allparams)

  # Don't plot if there are no pairs:
  if npars == 1:
    return

  # Set default parameter names:
  if parname is None:
    namelen = int(2+np.log10(npars-1))
    parname = np.zeros(npars, "|S%d"%namelen)
    for i in np.arange(npars):
      parname[i] = "P" + str(i).zfill(namelen-1)
  fs = 14

  # Set palette color:
  palette = plt.matplotlib.colors.LinearSegmentedColormap('YlOrRd2',
                                               plt.cm.datad['YlOrRd'], 65536)
  palette.set_under(alpha=0.0, color='w')
  #palette.set_under(color='w')

  plt.figure(fignum, figsize=(8,8))
  plt.clf()
  if title is not None:
    plt.suptitle(title, size=16)

  h = 1 # Subplot index
  plt.subplots_adjust(left=0.15,   right=0.95, bottom=0.15, top=0.9,
                      hspace=0.05, wspace=0.05)

  for   j in np.arange(1, npars): # Rows
    for i in np.arange(npars-1):  # Columns
      if j > i:
        a = plt.subplot(npars-1, npars-1, h)
        # Y labels:
        if i == 0:
          plt.yticks(size=fs)
          plt.ylabel(parname[j], size=fs, multialignment='center')
        else:
          a = plt.yticks(visible=False)
        # X labels:
        if j == npars-1:
          plt.xticks(size=fs, rotation=90)
          plt.xlabel(parname[i], size=fs)
        else:
          a = plt.xticks(visible=False)
        # The plot:
        if style=="hist":
          hist2d, xedges, yedges = np.histogram2d(allparams[i, 0::thinning],
                                     allparams[j, 0::thinning], 20, normed=True)
          vmin = np.min(hist2d[np.where(hist2d > 0)])
          a = plt.imshow(hist2d.T, extent=(xedges[0], xedges[-1], yedges[0],
                         yedges[-1]), cmap=palette, vmin=vmin, aspect='auto',
                         origin='lower', interpolation='bilinear')
        elif style=="points":
          a = plt.plot(allparams[i], allparams[j], ",")
      h += 1
  # The colorbar:
  if style == "hist":
    if npars > 2:
      a = plt.subplot(2, 6, 5, frameon=False)
      a.yaxis.set_visible(False)
      a.xaxis.set_visible(False)
    cb = plt.colorbar()
    cb.set_label("Normalized point density", fontsize=fs)
  # Save file:
  if savefile != None:
      plt.savefig(savefile)

def histogram(allparams, title=None, parname=None, thinning=1,
              fignum=-30, savefile=None):
  """
  Plot parameter marginal posterior distributions

  Parameters:
  -----------
  allparams: 2D ndarray
     An MCMC sampling array with dimension (number of parameters,
     sampling length).
  title: String
     Plot title.
  parname: Iterable (strings)
     List of label names for parameters.  If None use ['P0', 'P1', ...].
  thinning: Integer
     Thinning factor for plotting (plot every thinning-th value).
  fignum: Integer
     The figure number.
  savefile: Boolean
     If not None, name of file to save the plot.

  Modification History:
  ---------------------
  2007-2012?  kevin     Initial version by Kevin Stevenson, UCF.
  2014-04-19  patricio  Updated and documented. pcubillos@fulbrightmail.org
  """
  # Get number of parameters and length of chain:
  npars, niter = np.shape(allparams)
  fs = 14  # Fontsize

  # Set default parameter names:
  if parname is None:
    namelen = int(2+np.log10(npars-1))
    parname = np.zeros(npars, "|S%d"%namelen)
    for i in np.arange(npars):
      parname[i] = "P" + str(i).zfill(namelen-1)

  # Set number of rows:
  if npars < 10:
    nrows = (npars - 1)/3 + 1
  else:
    nrows = (npars - 1)/4 + 1
  # Set number of columns:
  if   npars > 9:
    ncolumns = 4
  elif npars > 4:
    ncolumns = 3
  else:
    ncolumns = (npars+2)/3 + (npars+2)%3  # (Trust me!)

  histheight = np.amin((2 + 2*(nrows), 8))
  if nrows == 1:
    bottom = 0.25
  else:
    bottom = 0.15

  plt.figure(fignum, figsize=(8, histheight))
  plt.clf()
  plt.subplots_adjust(left=0.1, right=0.95, bottom=bottom, top=0.9,
                      hspace=0.4, wspace=0.1)

  if title is not None:
    a = plt.suptitle(title, size=16)

  maxylim = 0  # Max Y limit
  for i in np.arange(npars):
    ax = plt.subplot(nrows, ncolumns, i+1)
    a  = plt.xticks(size=fs, rotation=90)
    if i%ncolumns == 0:
      a = plt.yticks(size=fs)
    else:
      a = plt.yticks(visible=False)      
    plt.xlabel(parname[i], size=fs)
    a = plt.hist(allparams[i,0::thinning], 20, normed=False)
    maxylim = np.amax((maxylim, ax.get_ylim()[1]))

  # Set uniform height:
  for i in np.arange(npars):
    ax = plt.subplot(nrows, ncolumns, i+1)
    ax.set_ylim(0, maxylim)

  if savefile != None:
    plt.savefig(savefile)

