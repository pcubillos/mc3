# Copyright (c) 2015-2018 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = ["trace", "pairwise", "histogram", "RMS", "modelfit", "subplotter"]

import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate as si

from .  import colormaps as cm
from .. import utils as mu

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../lib')
import binarray as ba

if sys.version_info.major == 2:
  range = xrange


def default_parnames(npars):
  namelen = len(str(npars))
  pnames = ["Param "]*npars
  for i in range(npars):
    pnames[i] += str(i+1).zfill(namelen)
  return pnames


def trace(posterior, Zchain=None, pnames=None, thinning=1,
          burnin=0, fignum=100, savefile=None, fmt=".", ms=2.5, fs=11):
  """
  Plot parameter trace MCMC sampling.

  Parameters
  ----------
  posterior: 2D float ndarray
     An MCMC posterior sampling with dimension: [nsamples, npars].
  Zchain: 1D integer ndarray
     the chain index for each posterior sample.
  pnames: Iterable (strings)
     Label names for parameters.
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
  ms: Float
     Marker size.
  fs: Float
     Fontsize of texts.

  Returns
  -------
  axes: 1D axes ndarray
     The array of axes containing the marginal posterior distributions.

  Uncredited Developers
  ---------------------
  Kevin Stevenson  (UCF)
  """
  # Get indices for samples considered in final analysis:
  if Zchain is not None:
    nchains = np.amax(Zchain) + 1
    good = np.zeros(len(Zchain), bool)
    for c in range(nchains):
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

  # Set default parameter names:
  if pnames is None:
    pnames = default_parnames(npars)

  npanels = 12  # Max number of panels per page
  npages = int(1 + (npars-1)/npanels)

  # Make the trace plot:
  axes = []
  i = 0
  for page in range(npages):
    fig = plt.figure(page, figsize=(8.5,11.0))
    plt.clf()
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.97,
                        hspace=0.15)
    while i < npars:
      ax = plt.subplot(npanels, 1, i%npanels+1)
      axes.append(ax)
      ax.plot(posterior[0::thinning,i], fmt, ms=ms)
      yran = ax.get_ylim()
      if Zchain is not None:
        ax.vlines(xsep, yran[0], yran[1], "0.5")
      # Y-axis adjustments:
      ax.set_ylim(yran)
      ax.locator_params(axis='y', nbins=5, tight=True)
      ax.tick_params(labelsize=fs-1)
      ax.set_ylabel(pnames[i], size=fs, multialignment='center')
      # X-axis adjustments:
      ax.set_xlim(0, xmax)
      ax.get_xaxis().set_visible(False)
      i += 1
      if i%npanels == 0:
        break
    ax.set_xlabel('MCMC sample', size=fs)
    ax.get_xaxis().set_visible(True)

    if savefile is not None:
      if npages > 1:
        sf = os.path.splitext(savefile)
        try:
          bbox = fig.get_tightbbox(fig._cachedRenderer).padded(0.1)
          bbox_points = bbox.get_points()
          bbox_points[:,0] = 0.0, 8.5
          bbox.set_points(bbox_points)
        except:  # May fail for ssh connection without X display
          ylow = 9.479 - 0.862*np.amin([npanels-1, npars-npanels*page-1])
          bbox = mpl.transforms.Bbox([[0.0, ylow], [8.5, 11]])

        fig.savefig("{:s}_page{:02d}{:s}".format(sf[0], page, sf[1]),
                    bbox_inches=bbox)
      else:
        fig.savefig(savefile, bbox_inches='tight')

  return axes


def pairwise(posterior, pnames=None, thinning=1, fignum=200,
             savefile=None, bestp=None, nbins=35, nlevels=20,
             absolute_dens=False, ranges=None, fs=11, rect=None, margin=0.01):
  """
  Plot parameter pairwise posterior distributions.

  Parameters
  ----------
  posterior: 2D ndarray
     An MCMC posterior sampling with dimension: [nsamples, nparameters].
  pnames: Iterable (strings)
     Label names for parameters.
  thinning: Integer
     Thinning factor for plotting (plot every thinning-th value).
  fignum: Integer
     The figure number.
  savefile: Boolean
     If not None, name of file to save the plot.
  bestp: 1D float ndarray
     If not None, plot the best-fitting values for each parameter
     given by bestp.
  nbins: Integer
     The number of grid bins for the 2D histograms.
  nlevels: Integer
     The number of contour color levels.
  ranges: List of 2-element arrays
     List with custom (lower,upper) x-ranges for each parameter.
     Leave None for default, e.g., ranges=[(1.0,2.0), None, (0, 1000)].
  fs: Float
     Fontsize of texts.
  rect: 1D list/ndarray
     If not None, plot the pairwise plots in current figure, within the
     ranges defined by rect (xleft, ybottom, xright, ytop).
  margin: Float
     Margins between panels (when rect is not None).

  Returns
  -------
  axes: 2D axes ndarray
     The grid of axes containing the pairwise posterior distributions.
  cb: axes
     The colorbar axes instance.

  Notes
  -----
  Note that rect delimits the boundaries of the panels. The labels and
  ticklabels will appear right outside rect, so the user needs to leave
  some wiggle room for them.

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

  if ranges is None:
    ranges = np.repeat(None, npars)
  else: # Set default ranges if necessary:
    for i in range(npars):
      if ranges[i] is None:
        ranges[i] = (np.nanmin(posterior[0::thinning,i]),
                     np.nanmax(posterior[0::thinning,i]))

  # Set default parameter names:
  if pnames is None:
    pnames = default_parnames(npars)

  # Set palette color:
  palette = cm.viridis_r
  palette.set_under(color='w')
  palette.set_bad(color='w')

  # Gather 2D histograms:
  hist = []
  xran, yran, lmax = [], [], []
  for   j in range(1, npars): # Rows
    for i in range(j):        # Columns
      ran = None
      if ranges[i] is not None:
        ran = [ranges[i], ranges[j]]
      h,x,y = np.histogram2d(posterior[0::thinning,i],
               posterior[0::thinning,j], bins=nbins, range=ran, normed=False)
      hist.append(h.T)
      xran.append(x)
      yran.append(y)
      lmax.append(np.amax(h)+1)
  # Reset upper boundary to absolute maximum value if requested:
  if absolute_dens:
    lmax = npars*(npars+1)*2 * [np.amax(lmax)]

  if rect is None:
    rect = (0.15, 0.15, 0.95, 0.95)
    plt.figure(fignum, figsize=(8,8))
    plt.clf()

  axes = np.tile(None, (npars-1, npars-1))
  # Plot:
  k = 0 # Histogram index
  for   j in range(1, npars): # Rows
    for i in range(j):        # Columns
      h = (npars-1)*(j-1) + i + 1  # Subplot index
      ax = axes[i,j-1] = subplotter(rect, margin, h, npars-1)
      # Labels:
      ax.tick_params(labelsize=fs-1)
      if i == 0:
        ax.set_ylabel(pnames[j], size=fs)
      else:
        ax.get_yaxis().set_visible(False)
      if j == npars-1:
        ax.set_xlabel(pnames[i], size=fs)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
      else:
        ax.get_xaxis().set_visible(False)
      # The plot:
      a = ax.contourf(hist[k], cmap=palette, vmin=1, origin='lower',
                  levels=[0]+list(np.linspace(1,lmax[k], nlevels)),
                  extent=(xran[k][0], xran[k][-1], yran[k][0], yran[k][-1]))
      for c in a.collections:
        c.set_edgecolor("face")
      if bestp is not None:
        ax.axvline(bestp[i], dashes=(6,4), color="0.5", lw=1.0)
        ax.axhline(bestp[j], dashes=(6,4), color="0.5", lw=1.0)
      if ranges[i] is not None:
        ax.set_xlim(ranges[i])
      if ranges[i] is not None:
        ax.set_ylim(ranges[j])
      k += 1

  # The colorbar:
  bounds = np.linspace(0, 1.0, nlevels)
  norm = mpl.colors.BoundaryNorm(bounds, palette.N)
  if rect is not None:
    dx = (rect[2]-rect[0])*0.05
    dy = (rect[3]-rect[1])*0.45
    ax2 = plt.axes([rect[2]-dx, rect[3]-dy, dx, dy])
  else:
    ax2 = plt.axes([0.85, 0.57, 0.025, 0.36])
  cb = mpl.colorbar.ColorbarBase(ax2, cmap=palette, norm=norm,
        spacing='proportional', boundaries=bounds, format='%.1f')
  cb.set_label("Posterior density", fontsize=fs)
  cb.ax.yaxis.set_ticks_position('left')
  cb.ax.yaxis.set_label_position('left')
  cb.ax.tick_params(labelsize=fs-1)
  cb.set_ticks(np.linspace(0, 1, 5))
  for c in ax2.collections:
    c.set_edgecolor("face")
  plt.draw()

  # Save file:
  if savefile is not None:
    plt.savefig(savefile)

  return axes, cb


def histogram(posterior, pnames=None, thinning=1, fignum=300,
              savefile=None, bestp=None, percentile=None, pdf=None,
              xpdf=None, ranges=None, axes=None, lw=2.0, fs=11):
  """
  Plot parameter marginal posterior distributions

  Parameters
  ----------
  posterior: 1D or 2D float ndarray
     An MCMC posterior sampling with dimension [nsamples] or
     [nsamples, nparameters].
  pnames: Iterable (strings)
     Label names for parameters.
  thinning: Integer
     Thinning factor for plotting (plot every thinning-th value).
  fignum: Integer
     The figure number.
  savefile: Boolean
     If not None, name of file to save the plot.
  bestp: 1D float ndarray
     If not None, plot the best-fitting values for each parameter
     given by bestp.
  percentile: Float
     If not None, plot the percentile- highest posterior density region
     of the distribution.  Note that this should actually be the
     fractional part, i.e. set percentile=0.68 for a 68% HPD.
  pdf: 1D float ndarray or list of ndarrays
     A smoothed PDF of the distribution for each parameter.
  xpdf: 1D float ndarray or list of ndarrays
     The X coordinates of the PDFs.
  ranges: List of 2-element arrays
     List with custom (lower,upper) x-ranges for each parameter.
     Leave None for default, e.g., ranges=[(1.0,2.0), None, (0, 1000)].
  axes: List of matplotlib.axes
     If not None, plot histograms in the currently existing axes.
  lw: Float
     Linewidth of the histogram contour.
  fs: Float
     Font size for texts.

  Returns
  -------
  axes: 1D axes ndarray
     The array of axes containing the marginal posterior distributions.

  Uncredited Developers
  ---------------------
  Kevin Stevenson  (UCF)
  """
  if np.ndim(posterior) == 1:
    posterior = np.expand_dims(posterior, axis=1)
  nsamples, npars = np.shape(posterior)

  if pdf is None: # Make list of Nones
    pdf  = [None]*npars
    xpdf = [None]*npars
  if not isinstance(pdf, list):  # Put single arrays into list
    pdf  = [pdf]
    xpdf = [xpdf]
  # Histogram keywords depending whether one wants the HPD or not:
  hkw = {'edgecolor':'navy', 'color':'b'}
  # Bestfit keywords:
  bkw = {'zorder':2, 'color':'orange'}
  if percentile is not None:
    hkw = {'histtype':'step', 'lw':lw, 'edgecolor':'b'}
    bkw = {'zorder':-1, 'color':'red'}

  # Set default parameter names:
  if pnames is None:
    pnames = default_parnames(npars)

  # Xranges:
  if ranges is None:
    ranges = np.repeat(None, npars)

  # Set number of rows:
  nrows, ncolumns, npanels = 4, 3, 12
  npages = int(1 + (npars-1)/npanels)

  if axes is None:
    newfig = True
    axes = []
  else:
    newfig = False
    npages = 1  # Assume there's only one page

  figs = np.tile(None, npages)
  maxylim = 0  # Max Y limit
  i = 0
  for j in range(npages):
    if newfig:
      figs[j] = plt.figure(fignum+j, figsize=(8.5, 11.0))
      plt.clf()
      plt.subplots_adjust(left=0.1, right=0.97, bottom=0.08, top=0.98,
                          hspace=0.5, wspace=0.1)
    else:
      figs[j] = axes[0].get_figure()

    while i < npars:
      if newfig:
        ax = plt.subplot(nrows, ncolumns, i%npanels+1)
        axes.append(ax)
        if i%ncolumns == 0:
          ax.set_ylabel(r"$N$ samples", fontsize=fs)
        else:
          ax.get_yaxis().set_visible(False)
      else:
        ax = axes[i]
        ax.get_yaxis().set_visible(False)  # No ylabel/yticklabels by default
      ax.tick_params(labelsize=fs-1)
      plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
      ax.set_xlabel(pnames[i], size=fs)
      vals, bins, h = ax.hist(posterior[0::thinning, i], bins=25,
                              range=ranges[i], normed=False, zorder=0, **hkw)
      # Plot HPD region:
      if percentile is not None:
        PDF, Xpdf, HPDmin = mu.credregion(posterior[:,i], percentile,
                                          pdf[i], xpdf[i])
        vals = np.r_[0, vals, 0]
        bins = np.r_[bins[0] - (bins[1]-bins[0]), bins]
        # interpolate xpdf into the histogram:
        f = si.interp1d(bins+0.5*(bins[1]-bins[0]), vals, kind='nearest')
        # Plot the HPD region as shaded areas:
        if ranges[i] is not None:
          xran = np.argwhere((Xpdf>ranges[i][0]) & (Xpdf<ranges[i][1]))
          Xpdf = Xpdf[np.amin(xran):np.amax(xran)]
          PDF  = PDF [np.amin(xran):np.amax(xran)]
        ax.fill_between(Xpdf, 0, f(Xpdf), where=PDF>=HPDmin,
           facecolor='0.75', edgecolor='none', interpolate=False, zorder=-2)
      if bestp is not None:
        ax.axvline(bestp[i], dashes=(7,4), lw=1.0, **bkw)
      maxylim = np.amax((maxylim, ax.get_ylim()[1]))
      i += 1
      if i%npanels == 0:
        break

  # Set uniform height and save:
  for ax in axes:
    ax.set_ylim(0, maxylim)

  # Save:
  if savefile is not None:
    for j in range(npages):
      if npages > 1:
        sf = os.path.splitext(savefile)
        figs[j].savefig("{:s}_page{:02d}{:s}".format(sf[0], j+1, sf[1]),
                    bbox_inches='tight')
      else:
        figs[j].savefig(savefile, bbox_inches='tight')

  return axes


def RMS(binsz, rms, stderr, rmslo, rmshi, cadence=None, binstep=1,
        timepoints=[], ratio=False, fignum=-40,
        yran=None, xran=None, savefile=None):
  """
  Plot the RMS vs binsize curve.

  Parameters
  ----------
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
    ylabel = r"$\beta$ = RMS / std error"
  else:
    ylabel = "RMS"

  plt.figure(fignum, (8,6))
  plt.clf()
  ax = plt.subplot(111)

  if ratio: # Plot the residuals-to-Gaussian RMS ratio:
    ax.errorbar(binsz[::binstep]*cadence, (rms/stderr)[::binstep],
                yerr=[(rmslo/stderr)[::binstep], (rmshi/stderr)[::binstep]],
                fmt='k-', ecolor='0.5', capsize=0, label="__nolabel__")
    ax.semilogx(xran, [1,1], "r-", lw=2)
  else:     # Plot residuals and Gaussian RMS individually:
    # Residuals RMS:
    ax.errorbar(binsz[::binstep]*cadence, rms[::binstep],
                yerr=[rmslo[::binstep], rmshi[::binstep]],
                fmt='k-', ecolor='0.5', capsize=0, label="RMS")
    # Gaussian noise projection:
    ax.loglog(binsz*cadence, stderr, color='red', ls='-',
                   lw=2, label="Gaussian std.")
    ax.legend(loc="best")
  for time in timepoints:
    ax.vlines(time, yran[0], yran[1], 'b', 'dashed', lw=2)

  ax.tick_params(labelsize=fs-1)
  ax.set_ylim(yran)
  ax.set_xlim(xran)
  ax.set_ylabel(ylabel, fontsize=fs)
  ax.set_xlabel(xlabel, fontsize=fs)

  if savefile is not None:
    plt.savefig(savefile)


def modelfit(data, uncert, indparams, model, nbins=75,
             fignum=-50, savefile=None, fmt="."):
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
  fs = 12 # Font-size

  plt.figure(fignum, figsize=(8,6))
  plt.clf()

  # Residuals:
  rax = plt.axes([0.15, 0.1, 0.8, 0.2])
  rax.errorbar(binindp, bindata-binmodel, binuncert, fmt='ko', ms=4)
  rax.plot([indparams[0], indparams[-1]], [0,0],'k:',lw=1.5)
  rax.tick_params(labelsize=fs-1)
  rax.set_xlabel("x", fontsize=fs)
  rax.set_ylabel('Residuals', fontsize=fs)

  # Data and Model:
  ax = plt.axes([0.15, 0.35, 0.8, 0.55])
  ax.errorbar(binindp, bindata, binuncert, fmt='ko', ms=4,
              label='Binned Data')
  ax.plot(indparams, model, "b", lw=2, label='Best Fit')
  ax.get_xaxis().set_visible(False)
  ax.tick_params(labelsize=fs-1)
  ax.set_ylabel('y', fontsize=fs)
  ax.legend(loc='best')

  if savefile is not None:
    plt.savefig(savefile)


def subplotter(rect, margin, ipan, nx, ny=None, ymargin=None):
  """
  Create an axis instance for one panel (with index ipan) of a grid
  of npanels, where the grid located inside rect (xleft, ybottom,
  xright, ytop).

  Parameters
  ----------
  rect: 1D List/ndarray
     Rectangle with xlo, ylo, xhi, yhi positions of the grid boundaries.
  margin: Float
     Width of margin between panels.
  ipan: Integer
     Index of panel to create (as in plt.subplots).
  nx: Integer
     Number of panels along the x axis.
  ny: Integer
     Number of panels along the y axis. If None, assume ny=nx.
  ymargin: Float
     Width of margin between panels along y axes (if None, adopt margin).

  Returns
  -------
  axes: axes instance
     A matplotlib axes instance at the specified position.
  """
  if ny is None:
    ny = nx
  if ymargin is None:
    ymargin = margin

  # Size of a panel:
  Dx = rect[2] - rect[0]
  Dy = rect[3] - rect[1]
  dx = Dx/nx - (nx-1.0)* margin/nx
  dy = Dy/ny - (ny-1.0)*ymargin/ny
  # Position of panel ipan:
  # Follow plt's scheme, where panel 1 is at the top left panel,
  # panel 2 is to the right of panel 1, and so on:
  xloc = (ipan-1) % nx
  yloc = (ny-1) - ((ipan-1) // nx)
  # Bottom-left corner of panel:
  xpanel = rect[0] + xloc*(dx+ margin)
  ypanel = rect[1] + yloc*(dy+ymargin)

  return plt.axes([xpanel, ypanel, dx, dy])
