# Copyright (c) 2015-2016 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ["trace", "pairwise", "histogram", "RMS", "modelfit"]

import sys, os
import numpy as np
import matplotlib as mpl
#mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../lib')
import binarray as ba


def trace(posterior, Zchain=None, title=None, parname=None, thinning=1,
          burnin=0, fignum=-10, savefile=None, fmt="."):
  """
  Plot parameter trace MCMC sampling

  Parameters
  ----------
  posterior: 2D float ndarray
     An MCMC posterior sampling with dimension: [nsamples, npars].
  Zchain: 1D integer ndarray
     the chain index for each posterior sample.
  title: String
     Plot title.
  parname: Iterable (strings)
     List of label names for parameters.  If None use ['P0', 'P1', ...].
  thinning: Integer
     Thinning factor for plotting (plot every thinning-th value).
  burnin: Integer
     Thinned burn-in number of iteration (only used when Zchain is not None).
  fignum: Integer
     The figure number.
  savefile: Boolean
     If not None, name of file to save the plot.
  fmt: String
     The format string for the line and marker.

  Uncredited Developers
  ---------------------
  Kevin Stevenson  (UCF)
  """

  # Get indices for samples considered in final analysis:
  if Zchain is not None:
    nchains = np.amax(Zchain) + 1
    good = np.zeros(len(Zchain), bool)
    for c in np.arange(nchains):
      good[np.where(Zchain == c)[0][burnin:]] = True
    # Values accepted for posterior stats:
    posterior = posterior[good]
    Zchain    = Zchain   [good]
    # Sort the posterior by chain:
    zsort = np.lexsort([Zchain])
    posterior = posterior[zsort]
    Zchain    = Zchain   [zsort]
    # Get location for chains separations:
    xsep = np.where(np.ediff1d(Zchain[0::thinning]))[0]

  # Get number of parameters and length of chain:
  nsamples, npars = np.shape(posterior)
  # Number of samples (thinned):
  xmax = len(posterior[0::thinning])
  fs = 14  # Fontsize

  # Set default parameter names:
  if parname is None:
    namelen = int(2+np.log10(np.amax([npars-1,1])))
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
    plt.plot(posterior[0::thinning, i], fmt)
    yran = a.get_ylim()
    if Zchain is not None:
      plt.vlines(xsep, yran[0], yran[1], "0.5")
    plt.xlim(0, xmax)
    plt.ylim(yran)
    plt.ylabel(parname[i], size=fs, multialignment='center')
    plt.yticks(size=fs)
    if i == npars - 1:
      plt.xticks(size=fs)
      plt.xlabel('MCMC sample', size=fs)
    else:
      plt.xticks(visible=False)

  if savefile is not None:
    plt.savefig(savefile)


def pairwise(posterior, title=None, parname=None, thinning=1,
             fignum=-11, savefile=None, style="hist"):
  """
  Plot parameter pairwise posterior distributions

  Parameters
  ----------
  posterior: 2D ndarray
     An MCMC posterior sampling with dimension: [nsamples, nparameters].
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

  Uncredited Developers
  ---------------------
  Kevin Stevenson  (UCF)
  Ryan Hardy       (UCF)
  """
  # Get number of parameters and length of chain:
  nsamples, npars = np.shape(posterior)

  # Don't plot if there are no pairs:
  if npars == 1:
    return

  # Set default parameter names:
  if parname is None:
    namelen = int(2+np.log10(np.amax([npars-1,1])))
    parname = np.zeros(npars, "|S%d"%namelen)
    for i in np.arange(npars):
      parname[i] = "P" + str(i).zfill(namelen-1)
  fs = 14

  # Set palette color:
  palette = plt.matplotlib.colors.LinearSegmentedColormap('YlOrRd2',
                                               plt.cm.datad['YlOrRd'], 256)
  palette.set_under(alpha=0.0)
  palette.set_bad(alpha=0.0)

  fig = plt.figure(fignum, figsize=(8,8))
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
          hist2d, xedges, yedges = np.histogram2d(posterior[0::thinning, i],
                                   posterior[0::thinning, j], 20, normed=False)
          vmin = 0.0
          hist2d[np.where(hist2d == 0)] = np.nan
          a = plt.imshow(hist2d.T, extent=(xedges[0], xedges[-1], yedges[0],
                         yedges[-1]), cmap=palette, vmin=vmin, aspect='auto',
                         origin='lower', interpolation='bilinear')
        elif style=="points":
          a = plt.plot(posterior[::thinning,i], posterior[::thinning,j], ",")
      h += 1
  # The colorbar:
  if style == "hist":
    if npars > 2:
      a = plt.subplot(2, 6, 5, frameon=False)
      a.yaxis.set_visible(False)
      a.xaxis.set_visible(False)
    bounds = np.linspace(0, 1.0, 64)
    norm = mpl.colors.BoundaryNorm(bounds, palette.N)
    ax2 = fig.add_axes([0.85, 0.535, 0.025, 0.36])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=palette, norm=norm,
          spacing='proportional', boundaries=bounds, format='%.1f')
    cb.set_label("Normalized point density", fontsize=fs)
    cb.set_ticks(np.linspace(0, 1, 5))
    plt.draw()

  # Save file:
  if savefile is not None:
    plt.savefig(savefile)


def histogram(posterior, title=None, parname=None, thinning=1,
              fignum=-12, savefile=None):
  """
  Plot parameter marginal posterior distributions

  Parameters
  ----------
  posterior: 2D float ndarray
     An MCMC posterior sampling with dimension: [nsamples, nparameters].
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

  Uncredited Developers
  ---------------------
  Kevin Stevenson  (UCF)
  """
  # Get number of parameters and length of chain:
  nsamples, npars = np.shape(posterior)
  fs = 14  # Fontsize

  # Set default parameter names:
  if parname is None:
    namelen = int(2+np.log10(np.amax([npars-1,1])))
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
    a = plt.hist(posterior[0::thinning, i], 20, normed=False)
    maxylim = np.amax((maxylim, ax.get_ylim()[1]))

  # Set uniform height:
  for i in np.arange(npars):
    ax = plt.subplot(nrows, ncolumns, i+1)
    ax.set_ylim(0, maxylim)

  if savefile is not None:
    plt.savefig(savefile)


def RMS(binsz, rms, stderr, rmslo, rmshi, cadence=None, binstep=1,
        timepoints=[], ratio=False, fignum=-20,
        yran=None, xran=None, savefile=None):
  """
  Plot the RMS vs binsize curve.

  Parameters:
  -----------
  binsz: 1D ndarray
     Array of bin sizes.
  rms: 1D ndarray
     RMS of dataset at given binsz.
  stderr: 1D ndarray
     Gaussian-noise rms Extrapolation
  rmslo: 1D ndarray
     RMS lower uncertainty
  rmshi: 1D ndarray
     RMS upper uncertainty
  cadence: Float
     Time between datapoints in seconds.
  binstep: Integer
     Plot every-binstep point.
  timepoints: List
     Plot a vertical line at each time-points.
  ratio: Boolean
     If True, plot rms/stderr, else, plot both curves.
  fignum: Integer
     Figure number
  yran: 2-elements tuple
     Minimum and Maximum y-axis ranges.
  xran: 2-elements tuple
     Minimum and Maximum x-axis ranges.
  savefile: String
     If not None, name of file to save the plot.
  """

  if np.size(rms) <= 1:
    return

  # Set cadence:
  if cadence is None:
    cadence = 1.0
    xlabel = "Bin size"
  else:
    xlabel = "Bin size  (sec)"

  # Set plotting limits:
  if yran is None:
    #yran = np.amin(rms), np.amax(rms)
    yran = [np.amin(rms-rmslo), np.amax(rms+rmshi)]
    yran[0] = np.amin([yran[0],stderr[-1]])
    if ratio:
      yran = [0, np.amax(rms/stderr) + 1.0]
  if xran is None:
    xran = [cadence, np.amax(binsz*cadence)]

  fs = 14 # Font size
  if ratio:
    ylabel = r"$\beta =$ RMS / std. error"
  else:
    ylabel = "RMS"

  plt.figure(fignum, (8,6))
  plt.clf()
  ax = plt.subplot(111)

  if ratio: # Plot the residuals-to-Gaussian RMS ratio:
    a = plt.errorbar(binsz[::binstep]*cadence, (rms/stderr)[::binstep],
                  yerr=[(rmslo/stderr)[::binstep], (rmshi/stderr)[::binstep]],
                  fmt='k-', ecolor='0.5', capsize=0, label="__nolabel__")
    a = plt.semilogx(xran, [1,1], "r-", lw=2)
  else:     # Plot residuals and Gaussian RMS individually:
    # Residuals RMS:
    a = plt.errorbar(binsz[::binstep]*cadence, rms[::binstep],
                     yerr=[rmslo[::binstep], rmshi[::binstep]],
                     fmt='k-', ecolor='0.5',
                     capsize=0, label="RMS")
    # Gaussian noise projection:
    a = plt.loglog(binsz*cadence, stderr, color='red', ls='-',
                   lw=2, label="Gaussian std.")
    a = plt.legend()
  for time in timepoints:
    a = plt.vlines(time, yran[0], yran[1], 'b', 'dashed', lw=2)

  a = plt.yticks(size=fs)
  a = plt.xticks(size=fs)
  a = plt.ylim(yran)
  a = plt.xlim(xran)
  a = plt.ylabel(ylabel, fontsize=fs)
  a = plt.xlabel(xlabel, fontsize=fs)

  if savefile is not None:
    plt.savefig(savefile)


def modelfit(data, uncert, indparams, model, nbins=75, title=None,
             fignum=-22, savefile=None, fmt="."):
  """
  Plot the binned dataset with given uncertainties and model curves
  as a function of indparams.
  In a lower panel, plot the residuals bewteen the data and model.

  Parameters
  ----------
  data:  1D float ndarray
    Input data set.
  uncert:  1D float ndarray
    One-sigma uncertainties of the data points.
  indparams:  1D float ndarray
    Independent variable (X axis) of the data points.
  model:  1D float ndarray
    Model of data.
  nbins:  Integer
    Number of bins in the output plot.
  title:  String
    If not None, plot title.
  fignum:  Integer
    The figure number.
  savefile:  Boolean
    If not None, name of file to save the plot.
  fmt:  String
    Format of the plotted markers.
  """

  # Bin down array:
  binsize = int((np.size(data)-1)/nbins + 1)
  bindata, binuncert, binindp = ba.binarray(data, uncert, indparams, binsize)
  binmodel = ba.weightedbin(model, binsize)
  fs = 14 # Font-size

  p = plt.figure(fignum, figsize=(8,6))
  p = plt.clf()

  # Residuals:
  a = plt.axes([0.15, 0.1, 0.8, 0.2])
  p = plt.errorbar(binindp, bindata-binmodel, binuncert, fmt='ko', ms=4)
  p = plt.plot([indparams[0], indparams[-1]], [0,0],'k:',lw=1.5)
  p = plt.xticks(size=fs)
  p = plt.yticks(size=fs)
  p = plt.xlabel("x", size=fs)
  p = plt.ylabel('Residuals', size=fs)

  # Data and Model:
  a = plt.axes([0.15, 0.35, 0.8, 0.55])
  if title is not None:
    p = plt.title(title, size=fs)
  p = plt.errorbar(binindp, bindata, binuncert, fmt='ko', ms=4,
                   label='Binned Data')
  p = plt.plot(indparams, model, "b", lw=2, label='Best Fit')
  p = plt.setp(a.get_xticklabels(), visible = False)
  p = plt.yticks(size=13)
  p = plt.ylabel('y', size=fs)
  p = plt.legend(loc='best')

  if savefile is not None:
      p = plt.savefig(savefile)
