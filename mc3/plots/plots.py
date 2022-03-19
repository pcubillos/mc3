# Copyright (c) 2015-2022 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = [
    'trace',
    'histogram',
    'pairwise',
    'rms',
    'modelfit',
    'subplotter',
    'themes',
    'subplot',
    # Objects:
    'Posterior',
]

import os
import sys
import copy

import numpy as np
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import scipy.interpolate as si

from .. import utils as mu
from .. import stats as ms


# Color themes for histogram plots:
themes = {
    'default':{
        'edgecolor':'blue',
        'facecolor':'royalblue',
        'color':'navy',
        'colormap': plt.cm.viridis_r,
        },
    'blue':{
        'edgecolor':'blue',
        'facecolor':'royalblue',
        'color':'navy',
        'colormap': plt.cm.Blues,
        },
    'red': {
        'edgecolor':'crimson',
        'facecolor':'orangered',
        'color':'darkred',
        'colormap': plt.cm.Reds,
        },
    'black':{
        'edgecolor':'0.3',
        'facecolor':'0.3',
        'color':'black',
        'colormap': plt.cm.Greys,
        },
    'green':{
        'edgecolor':'forestgreen',
        'facecolor':'limegreen',
        'color':'darkgreen',
        'colormap': plt.cm.YlGn,
        },
    'orange':{
        'edgecolor':'darkorange',
        'facecolor':'gold',
        'color':'darkgoldenrod',
        'colormap': plt.cm.YlOrBr,
        },
    }


def trace(posterior, zchain=None, pnames=None, thinning=1,
    burnin=0, fignum=1000, savefile=None, fmt=".", ms=2.5, fs=11):
    """
    Plot parameter trace MCMC sampling.

    Parameters
    ----------
    posterior: 2D float ndarray
        An MCMC posterior sampling with dimension: [nsamples, npars].
    zchain: 1D integer ndarray
        the chain index for each posterior sample.
    pnames: Iterable (strings)
        Label names for parameters.
    thinning: Integer
        Thinning factor for plotting (plot every thinning-th value).
    burnin: Integer
        Thinned burn-in number of iteration (only used when zchain is not None).
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
    axes: 1D list of matplotlib.axes.Axes
        List of axes containing the marginal posterior distributions.
    """
    # Get indices for samples considered in final analysis:
    if zchain is not None:
        nchains = np.amax(zchain) + 1
        good = np.zeros(len(zchain), bool)
        for c in range(nchains):
            good[np.where(zchain == c)[0][burnin:]] = True
        # Values accepted for posterior stats:
        posterior = posterior[good]
        zchain    = zchain   [good]
        # Sort the posterior by chain:
        zsort = np.lexsort([zchain])
        posterior = posterior[zsort]
        zchain    = zchain   [zsort]
        # Get location for chains separations:
        xsep = np.where(np.ediff1d(zchain[0::thinning]))[0]

    # Get number of parameters and length of chain:
    nsamples, npars = np.shape(posterior)
    # Number of samples (thinned):
    xmax = len(posterior[0::thinning])

    # Set default parameter names:
    if pnames is None:
        pnames = mu.default_parnames(npars)

    npanels = 12  # Max number of panels per page
    npages = int(1 + (npars-1)/npanels)

    # Make the trace plot:
    axes = []
    ipar = 0
    for page in range(npages):
        fig = plt.figure(fignum+page, figsize=(8.5,11.0))
        plt.clf()
        plt.subplots_adjust(
            left=0.15, right=0.95, bottom=0.05, top=0.97, hspace=0.15)
        while ipar < npars:
            ax = plt.subplot(npanels, 1, ipar%npanels+1)
            axes.append(ax)
            ax.plot(posterior[0::thinning,ipar], fmt, ms=ms)
            yran = ax.get_ylim()
            if zchain is not None:
                ax.vlines(xsep, yran[0], yran[1], "0.5")
            # Y-axis adjustments:
            ax.set_ylim(yran)
            ax.locator_params(axis='y', nbins=5, tight=True)
            ax.tick_params(labelsize=fs-1, direction='in', top=True, right=True)
            ax.set_ylabel(pnames[ipar], size=fs, multialignment='center')
            # X-axis adjustments:
            ax.set_xlim(0, xmax)
            ax.get_xaxis().set_visible(False)
            ipar += 1
            if ipar%npanels == 0:
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
                    ylow = 9.479-0.862*np.amin([npanels-1,npars-npanels*page-1])
                    bbox = mpl.transforms.Bbox([[0.0, ylow], [8.5, 11]])

                fig.savefig(f"{sf[0]}_page{page:02d}{sf[1]}", bbox_inches=bbox)
            else:
                fig.savefig(savefile, bbox_inches='tight')

    return axes


def histogram(posterior, pnames=None, thinning=1, fignum=1100,
    savefile=None, bestp=None, quantile=None, pdf=None,
    xpdf=None, ranges=None, axes=None, lw=2.0, fs=11,
    nbins=25, theme='blue', yscale=False, orientation='vertical'):
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
    quantile: Float
        If not None, plot the quantile- highest posterior density region
        of the distribution.  For example, set quantile=0.68 for a 68% HPD.
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
    nbins: Integer
        The number of histogram bins.
    theme: String or dict
        The histograms' color theme.  If string must be one of mc3.plots.themes.
        If dict, must define edgecolor, facecolor, color (with valid matplotlib
        colors) for the histogram edge and face colors, and the best-fit color,
        respectively.
    yscale: Bool
        If True, set an absolute Y-axis scaling among all posteriors.
        Defaulted to False.
    orientation: String
        Orientation of the histograms.  If 'horizontal', the bottom of the
        histogram will be at the left (might require some adjusting of the
        axes location, e.g., a plt.tight_layout() call).

    Returns
    -------
    axes: 1D list of matplotlib.axes.Axes
        List of axes containing the marginal posterior distributions.
    """
    if isinstance(theme, str):
        theme = themes[theme]

    if np.ndim(posterior) == 1:
        posterior = np.expand_dims(posterior, axis=1)
    nsamples, npars = np.shape(posterior)

    if pdf is None:
        pdf  = [None]*npars
        xpdf = [None]*npars
    if not isinstance(pdf, list):  # Put single arrays into list
        pdf  = [pdf]
        xpdf = [xpdf]
    # Histogram keywords:
    if int(np.__version__.split('.')[1]) >= 15:
        hkw = {'density':not yscale}
    else:
        hkw = {'normed':not yscale}

    # Set default parameter names:
    if pnames is None:
        pnames = mu.default_parnames(npars)

    # Xranges:
    if ranges is None:
        ranges = np.repeat(None, npars)

    # Set number of rows:
    nrows, ncolumns, npanels = 4, 3, 12
    npages = int(1 + (npars-1)/npanels)

    ylabel = "$N$ samples" if yscale else "Posterior density"
    if axes is None:
        figs = []
        axes = []
        for j in range(npages):
            fig = plt.figure(fignum+j, figsize=(8.5, 11.0))
            figs.append(fig)
            fig.clf()
            fig.subplots_adjust(
                left=0.1, right=0.97, bottom=0.08, top=0.98,
                hspace=0.5, wspace=0.1)
            for ipar in range(np.amin([npanels, npars-npanels*j])):
                ax = fig.add_subplot(nrows, ncolumns, ipar+1)
                axes.append(ax)
                yax = ax.yaxis if orientation == 'vertical' else ax.xaxis
                if ipar%ncolumns == 0 or orientation == 'horizontal':
                    yax.set_label_text(ylabel, fontsize=fs)
                if ipar%ncolumns != 0 or yscale is False:
                    yax.set_ticklabels([])
    else:
        npages = 1  # Assume there's only one page
        figs = [axes[0].get_figure()]
        for ax in axes:
            ax.set_yticklabels([])

    maxylim = 0
    for ipar in range(npars):
        ax = axes[ipar]
        if orientation == 'vertical':
            xax = ax.xaxis
            get_xlim, set_xlim = ax.get_xlim, ax.set_xlim
            get_ylim, set_ylim = ax.get_ylim, ax.set_ylim
            fill_between = ax.fill_between
            axline = ax.axvline
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        else:
            xax = ax.yaxis
            get_xlim, set_xlim = ax.get_ylim, ax.set_ylim
            get_ylim, set_ylim = ax.get_xlim, ax.set_xlim
            fill_between = ax.fill_betweenx
            axline = ax.axhline

        ax.tick_params(labelsize=fs-1, direction='in', top=True, right=True)
        xax.set_label_text(pnames[ipar], fontsize=fs)
        vals, bins, h = ax.hist(
            posterior[0::thinning,ipar],
            bins=nbins, histtype='step', lw=lw, zorder=0,
            range=ranges[ipar], ec=theme['edgecolor'],
            orientation=orientation, **hkw)
        # Plot HPD region if needed:
        if quantile is None:
            ax.hist(
                posterior[0::thinning,ipar],
                bins=nbins, lw=lw, zorder=-2, alpha=0.4,
                range=ranges[ipar], facecolor=theme['facecolor'], ec='none',
                orientation=orientation, **hkw)
        if quantile is not None:
            PDF, Xpdf, HPDmin = ms.cred_region(
                posterior[:,ipar], quantile, pdf[ipar], xpdf[ipar])
            vals = np.r_[0, vals, 0]
            bins = np.r_[bins[0] - (bins[1]-bins[0]), bins]
            # Interpolate xpdf into the histogram:
            f = si.interp1d(bins+0.5*(bins[1]-bins[0]), vals, kind='nearest')
            # Plot the HPD region as shaded areas:
            if ranges[ipar] is not None:
                xran = np.argwhere((Xpdf>ranges[ipar][0])
                                 & (Xpdf<ranges[ipar][1]))
                Xpdf = Xpdf[np.amin(xran):np.amax(xran)]
                PDF  = PDF [np.amin(xran):np.amax(xran)]
            fill_between(
                Xpdf, 0, f(Xpdf), where=PDF>=HPDmin,
                facecolor=theme['facecolor'], edgecolor='none',
                interpolate=False, zorder=-2, alpha=0.4)
        if bestp is not None:
            axline(bestp[ipar], dashes=(7,4), lw=1.25, color=theme['color'])
        maxylim = np.amax((maxylim, get_ylim()[1]))
        if ranges[ipar] is not None:
            set_xlim(np.clip(get_xlim(), ranges[ipar][0], ranges[ipar][1]))

    if yscale:
        for ax in axes:
            set_ylim = ax.get_ylim if orientation == 'vertical' else ax.set_xlim
            set_ylim(0, maxylim)

    if savefile is not None:
        if npages == 1:
            savefiles = [savefile]
        else:
            root, ext = os.path.splitext(savefile)
            savefiles = [
                f"{root}_page{page:02d}{ext}" for page in range(figs)]
        for savefile, fig in zip(savefiles, figs):
            fig.savefig(savefile, bbox_inches='tight')

    return axes


def pairwise(posterior, pnames=None, thinning=1, fignum=1200,
    savefile=None, bestp=None, nbins=25, nlevels=20,
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
    axes: 2D ndarray of matplotlib.axes.Axes
        Array of axes containing the marginal posterior distributions.
    cb: matplotlib.axes.Axes
        The colorbar axes.

    Notes
    -----
    rect delimits the boundaries of the panels. The labels and
    ticklabels will appear outside rect, so the user needs to leave
    some wiggle room for them.
    """
    # Get number of parameters and length of chain:
    nsamples, npars = np.shape(posterior)

    # Don't plot if there are no pairs:
    if npars == 1:
        return None, None

    if ranges is None:
        ranges = np.repeat(None, npars)
    else: # Set default ranges if necessary:
        for i in range(npars):
            if ranges[i] is None:
                ranges[i] = (
                    np.nanmin(posterior[0::thinning,i]),
                    np.nanmax(posterior[0::thinning,i]))

    # Set default parameter names:
    if pnames is None:
        pnames = mu.default_parnames(npars)

    # Set palette color:
    palette = copy.copy(plt.cm.viridis_r)
    palette.set_under(color='w')
    palette.set_bad(color='w')

    # Gather 2D histograms:
    hist = []
    xran, yran, lmax = [], [], []
    for irow in range(1, npars):
        for icol in range(irow):
            ran = None
            if ranges[icol] is not None:
                ran = [ranges[icol], ranges[irow]]
            h, x, y = np.histogram2d(
                posterior[0::thinning,icol], posterior[0::thinning,irow],
                bins=nbins, range=ran, density=False)
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
    for irow in range(1, npars):
        for icol in range(irow):
            h = (npars-1)*(irow-1) + icol + 1  # Subplot index
            ax = axes[icol,irow-1] = subplotter(rect, margin, h, npars-1)
            # Labels:
            ax.tick_params(labelsize=fs-1, direction='in')
            if icol == 0:
                ax.set_ylabel(pnames[irow], size=fs)
            else:
                ax.set_yticklabels([])
            if irow == npars-1:
                ax.set_xlabel(pnames[icol], size=fs)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
            else:
                ax.set_xticklabels([])
            # The plot:
            cont = ax.contourf(
                hist[k], cmap=palette, vmin=1, origin='lower',
                levels=[0]+list(np.linspace(1,lmax[k], nlevels)),
                extent=(xran[k][0], xran[k][-1], yran[k][0], yran[k][-1]))
            for c in cont.collections:
                c.set_edgecolor("face")
            if bestp is not None:
                ax.axvline(bestp[icol], dashes=(6,4), color="0.5", lw=1.0)
                ax.axhline(bestp[irow], dashes=(6,4), color="0.5", lw=1.0)
            if ranges[icol] is not None:
                ax.set_xlim(ranges[icol])
            if ranges[icol] is not None:
                ax.set_ylim(ranges[irow])
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
    cb = mpl.colorbar.ColorbarBase(
        ax2, cmap=palette, norm=norm,
        spacing='proportional', boundaries=bounds, format='%.1f')
    cb.set_label("Posterior density", fontsize=fs)
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.yaxis.set_label_position('left')
    cb.ax.tick_params(labelsize=fs-1, direction='in', top=True, right=True)
    cb.set_ticks(np.linspace(0, 1, 5))
    for c in ax2.collections:
        c.set_edgecolor("face")
    plt.draw()

    # Save file:
    if savefile is not None:
        plt.savefig(savefile)

    return axes, cb


def rms(binsz, rms, stderr, rmslo, rmshi, cadence=None, binstep=1,
    timepoints=[], ratio=False, fignum=1300,
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

    Returns
    -------
    ax: matplotlib.axes.Axes
        Axes instance containing the marginal posterior distributions.
    """
    if cadence is None:
        cadence = 1.0
        xlabel = "Bin size"
    else:
        xlabel = "Bin size (sec)"

    if yran is None:
        yran = [np.amin(rms-rmslo), np.amax(rms+rmshi)]
        yran[0] = np.amin([yran[0],stderr[-1]])
        if ratio:
            yran = [0, np.amax(rms/stderr) + 1.0]
    if xran is None:
        xran = [cadence, np.amax(binsz*cadence)]

    fs = 14  # Font size
    ylabel = r"$\beta$ = RMS / std error" if ratio else "RMS"

    plt.figure(fignum, (8,6))
    plt.clf()
    ax = plt.subplot(111)

    if ratio:
        ax.errorbar(
            binsz[::binstep]*cadence, (rms/stderr)[::binstep],
            yerr=[(rmslo/stderr)[::binstep], (rmshi/stderr)[::binstep]],
            fmt='k-', ecolor='0.5', capsize=0, label="__nolabel__")
        ax.semilogx(xran, [1,1], "r-", lw=2)
    else:
        # Residuals RMS:
        ax.errorbar(
            binsz[::binstep]*cadence, rms[::binstep],
            yerr=[rmslo[::binstep], rmshi[::binstep]],
            fmt='k-', ecolor='0.5', capsize=0, label="RMS")
        # Gaussian noise projection:
        ax.loglog(
            binsz*cadence, stderr, color='red', ls='-', lw=2,
            label="Gaussian std.")
        ax.legend(loc="best")

    for time in timepoints:
        ax.vlines(time, yran[0], yran[1], 'b', 'dashed', lw=2)

    ax.tick_params(labelsize=fs-1, direction='in', top=True, right=True)
    ax.set_ylim(yran)
    ax.set_xlim(xran)
    ax.set_ylabel(ylabel, fontsize=fs)
    ax.set_xlabel(xlabel, fontsize=fs)

    if savefile is not None:
        plt.savefig(savefile)
    return ax


def modelfit(data, uncert, indparams, model, nbins=75,
    fignum=1400, savefile=None, fmt="."):
    """
    Plot the binned dataset with given uncertainties and model curves
    as a function of indparams.
    In a lower panel, plot the residuals bewteen the data and model.

    Parameters
    ----------
    data: 1D float ndarray
        Input data set.
    uncert: 1D float ndarray
        One-sigma uncertainties of the data points.
    indparams: 1D float ndarray
        Independent variable (X axis) of the data points.
    model: 1D float ndarray
        Model of data.
    nbins: Integer
        Number of bins in the output plot.
    fignum: Integer
        The figure number.
    savefile: Boolean
        If not None, name of file to save the plot.
    fmt: String
        Format of the plotted markers.

    Returns
    -------
    ax: matplotlib.axes.Axes
        Axes instance containing the marginal posterior distributions.
    """
    # Bin down array:
    binsize = int((np.size(data)-1)/nbins + 1)
    binindp  = ms.bin_array(indparams, binsize)
    binmodel = ms.bin_array(model,     binsize)
    bindata, binuncert = ms.bin_array(data, binsize, uncert)
    fs = 12 # Font-size

    plt.figure(fignum, figsize=(8,6))
    plt.clf()

    # Residuals:
    rax = plt.axes([0.15, 0.1, 0.8, 0.2])
    rax.errorbar(binindp, bindata-binmodel, binuncert, fmt='ko', ms=4)
    rax.plot([indparams[0], indparams[-1]], [0,0],'k:',lw=1.5)
    rax.tick_params(labelsize=fs-1, direction='in', top=True, right=True)
    rax.set_xlabel("x", fontsize=fs)
    rax.set_ylabel('Residuals', fontsize=fs)

    # Data and Model:
    ax = plt.axes([0.15, 0.35, 0.8, 0.55])
    ax.errorbar(
        binindp, bindata, binuncert, fmt='ko', ms=4, label='Binned data')
    ax.plot(indparams, model, "b", lw=2, label='Best Fit')
    ax.set_xticklabels([])
    ax.tick_params(labelsize=fs-1, direction='in', top=True, right=True)
    ax.set_ylabel('y', fontsize=fs)
    ax.legend(loc='best')

    if savefile is not None:
        plt.savefig(savefile)
    return ax, rax


def subplotter(rect, margin, ipan, nx, ny=None, ymargin=None):
    # TBD: Deprecate warning
    return subplot(rect, margin, ipan, nx, ny, ymargin)


def subplot(rect, margin, pos, nx, ny=None, ymargin=None, dry=False):
    """
    Create an axis instance for one panel (with index pos) of a grid
    of npanels, where the grid located inside rect (xleft, ybottom,
    xright, ytop).

    Parameters
    ----------
    rect: 1D List/ndarray
        Rectangle with xlo, ylo, xhi, yhi positions of the grid boundaries.
    margin: Float
        Width of margin between panels.
    pos: Integer
        Index of panel to create (as in plt.subplots).
    nx: Integer
        Number of panels along the x axis.
    ny: Integer
        Number of panels along the y axis. If None, assume ny=nx.
    ymargin: Float
        Width of margin between panels along y axes (if None, adopt margin).

    Returns
    -------
    axes: Matplotlib.axes.Axes
        An Axes instance at the specified position.
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
    # Position of panel pos:
    # Follow plt's scheme, where panel 1 is at the top left panel,
    # panel 2 is to the right of panel 1, and so on:
    xloc = (pos-1) % nx
    yloc = (ny-1) - ((pos-1) // nx)
    # Bottom-left corner of panel:
    xpanel = rect[0] + xloc*(dx+ margin)
    ypanel = rect[1] + yloc*(dy+ymargin)

    if dry:
        return [xpanel, ypanel, dx, dy]
    return plt.axes([xpanel, ypanel, dx, dy])


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def _histogram(posterior, pnames, bestp, ranges, axes,
    nbins, quantile, pdf, xpdf,
    linewidth, fontsize, theme, yscale, orientation):
    """
    Lowest-lever routine to plot marginal posterior distributions.
    >>> posterior = self.posterior
    >>> pnames = self.hist_pnames
    >>> bestp = self.bestp
    >>> ranges = self.ranges
    >>> axes = self.hist_axes
    >>> ticklabels = [ax is axes[-1] for ax in axes]
    >>> ticklabels = [pname != '' for pname in pnames]
    >>> yscale = False
    >>> orientation = 'vertical'
    >>> linewidth = self.lw
    >>> fontsize = self.fontsize
    >>> quantile = self.quantile
    >>> nbins = self.bins
    >>> theme = self.theme
    """
    nsamples, npars = np.shape(posterior)

    # Put all other keywords here?
    hist_kw = {
        'bins': nbins,
        'linewidth': linewidth,
        'orientation': orientation,
        'facecolor': to_rgba(theme['facecolor'], alpha=0.6),
        'edgecolor': theme['edgecolor'],
        'histtype': 'stepfilled',
        'density': not yscale,
    }
    if quantile is not None:
        hist_kw['facecolor'] = 'none'


    maxylim = 0
    for i in range(npars):
        ax = axes[i]
        ax.clear()  # For testing only
        if orientation == 'vertical':
            xax, yax = ax.xaxis, ax.yaxis
            fill_between = ax.fill_between
            axline = ax.axvline
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        else:
            xax, yax = ax.yaxis, ax.xaxis
            fill_between = ax.fill_betweenx
            axline = ax.axhline

        ax.tick_params(labelsize=fontsize-1, direction='in', top=True)
        xax.set_label_text(pnames[i], fontsize=fontsize)
        vals, bins, h = ax.hist(
            posterior[:,i], range=ranges[i], **hist_kw)
        # Plot the HPD region as shaded areas:
        if quantile is not None:
            PDF, Xpdf, hpd_min = ms.cred_region(
                posterior[:,i], quantile, pdf[i], xpdf[i])
            vals = np.r_[0, vals, 0]
            bins = np.r_[bins[0] - (bins[1]-bins[0]), bins]
            f = si.interp1d(bins+0.5*(bins[1]-bins[0]), vals, kind='nearest')
            xran = (xpdf[i]>ranges[i][0]) & (xpdf[i]<ranges[i][1])
            fill_between(
                xpdf[i][xran], 0, f(xpdf[i][xran]), where=pdf[i][xran]>=hpd_min,
                facecolor=theme['facecolor'], edgecolor='none',
                interpolate=False,
                alpha=0.6)

        if bestp[i] is not None:
            axline(bestp[i], dashes=(7,4), lw=1.25, color=theme['color'])
        maxylim = np.amax((maxylim, yax.get_view_interval()[1]))
        xax.set_view_interval(*ranges[i], ignore=True)
        if pnames[i] == '':
            xax.set_ticklabels([])
        yax.set_ticklabels([])

    if yscale:
        for ax in axes:
            yax = ax.yaxis if orientation=='vertical' else ax.xaxis
            yax.set_view_interval(0, maxylim, ignore=True)


def _pairwise(posterior, pnames, bestp, ranges, axes,
    nbins, nlevels, absolute_dens=False,
    palette=None, fontsize=11, rect=None,
    hist_xran=None, hist=None, lmax=None):
    """
    Lowest-lever routine to plot pair-wise posterior distributions.
    >>> posterior = self.posterior
    >>> pnames = self.pnames
    >>> bestp = self.bestp
    >>> ranges = self.ranges
    >>> axes = self.pair_axes
    >>> fontsize = self.fontsize
    >>> nlevels = self.nlevels
    >>> rect = self.rect
    >>> nbins = self.bins
    >>> absolute_dens = False
    """
    # Get number of parameters and length of chain:
    nsamples, npars = np.shape(posterior)

    # Reset upper boundary to absolute maximum value if requested:
    if absolute_dens:
        lmax[:] = np.amax(lmax)

    for icol in range(npars-1):
        for irow in range(icol, npars-1):
            ax = axes[irow,icol]
            ax.clear()
            # Labels:
            ax.tick_params(labelsize=fontsize-1, direction='in')
            if icol == 0:
                ax.set_ylabel(pnames[irow+1], size=fontsize)
            else:
                ax.set_yticklabels([])
            if irow == npars-2:
                ax.set_xlabel(pnames[icol], size=fontsize)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
            else:
                ax.set_xticklabels([])
            # The plot:
            cont = ax.contourf(
                hist[irow,icol], cmap=palette, vmin=1, origin='lower',
                levels=[0]+list(np.linspace(1,lmax[irow,icol], nlevels)),
                extent=(hist_xran[icol,0], hist_xran[icol,-1],
                        hist_xran[irow+1,0], hist_xran[irow+1,-1]))
            for c in cont.collections:
                c.set_edgecolor("face")
            if bestp[icol] is not None:
                ax.axvline(bestp[icol], dashes=(6,4), color="0.5", lw=1.0)
            if bestp[irow+1] is not None:
                ax.axhline(bestp[irow+1], dashes=(6,4), color="0.5", lw=1.0)
            if ranges[icol] is not None:
                ax.set_xlim(ranges[icol])
            if ranges[icol] is not None:
                ax.set_ylim(ranges[irow+1])


def hist_2D(posterior, ranges, nbins, nlevels):
    """Construct 2D histograms."""
    nsamples, npars = np.shape(posterior)
    # Column index matches par index, row index matches par index + 1
    hist_xran = np.zeros((npars, nbins+1))
    hist = np.zeros((npars-1, npars-1, nbins, nbins))
    lmax = np.zeros((npars-1, npars-1))
    for icol in range(npars-1):
        for irow in range(icol, npars-1):
            ran = None
            if ranges[icol] is not None:
                ran = [ranges[irow+1], ranges[icol]]
            h, y, x = np.histogram2d(
                posterior[:,irow+1], posterior[:,icol], bins=nbins,
                range=ran, density=False)
            hist[irow, icol] = h
            if icol == 0:
                hist_xran[irow+1] = y
            if irow == 0 and icol == 0:
                hist_xran[irow] = x
            lmax[irow, icol] = np.amax(h) + 1
    return hist_xran, hist, lmax


class SoftUpdate:
    """ https://docs.python.org/3/howto/descriptor.html """
    def __set_name__(self, obj, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        value = getattr(obj, self.private_name)
        return value

    def __set__(self, obj, value):
        # TBD: Delete when done:
        print(f'Updating {self.private_name[1:]} to {value}')
        setattr(obj, self.private_name, value)
        if obj.pair_axes is not None:
            nx = obj.npars - int(not obj.plot_marginal)
            for icol in range(obj.npars-1):
                for irow in range(icol, obj.npars-1):
                    ax = obj.pair_axes[irow,icol]
                    h = nx*irow + icol + 1 + obj.npars*int(obj.plot_marginal)
                    ax.set_position(subplot(
                        obj.rect, obj.margin, h, nx, nx, obj.ymargin, dry=True))

                    ax.tick_params(labelsize=obj.fontsize-1, direction='in')
                    if icol == 0:
                        ax.set_ylabel(obj.pnames[irow+1], size=obj.fontsize)
                    else:
                        ax.set_yticklabels([])
                    if irow == obj.npars-2:
                        ax.set_xlabel(obj.pnames[icol], size=obj.fontsize)
                        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
                    else:
                        ax.set_xticklabels([])

        if obj.hist_axes is not None:
            obj.hist_pnames = [
                '' if i < obj.npars-1 else obj.pnames[i]
                for i in range(obj.npars)]

            for i in range(obj.npars):
                ax = obj.hist_axes[i]
                ax.set_visible(obj.plot_marginal)
                nx = obj.npars
                h = (obj.npars+1)*i + 1
                ax.set_position(subplot(
                    obj.rect, obj.margin, h, nx, nx, obj.ymargin, dry=True))

                if obj.orientation == 'vertical':
                    xax = ax.xaxis
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
                else:
                    xax = ax.yaxis
                ax.tick_params(
                    labelsize=obj.fontsize-1, direction='in', top=True)
                xax.set_label_text(obj.hist_pnames[i], fontsize=obj.fontsize)

        # if colorbar is not None:
        #     pass


class SizeUpdate(SoftUpdate):
    def __set__(self, obj, value):
        print(f'Updating {self.private_name[1:]} to {value}')
        setattr(obj, self.private_name, tuple(value))
        if obj.fig is not None:
            obj.fig.set_size_inches(*list(value))


class ThinningUpdate(SoftUpdate):
    def __set__(self, obj, value):
        if not hasattr(obj, 'input_posterior'):
            return
        print(f'Updating {self.private_name[1:]} to {value}')
        setattr(obj, self.private_name, value)
        obj.posterior = obj.input_posterior[0::value]


class ThemeUpdate(SoftUpdate):
    def __set__(self, obj, value):
        print(f'Updating {self.private_name[1:]} to {value}')
        if isinstance(value, str):
            value = themes[value]
        setattr(obj, self.private_name, value)


class BestpUpdate(SoftUpdate):
    def __set__(self, obj, value):
        if not hasattr(obj, 'npars'):
            return
        print(f'Updating {self.private_name[1:]} to {value}')
        if value is None:
            value = [None for _ in range(obj.npars)]
        if len(value) != obj.npars:
            raise ValueError(
                f"Invalid {self.private_name[1:]} input. Array size "
                f"({len(value)}) does not match number of parameters "
                f"({obj.npars})")
        setattr(obj, self.private_name, value)


class RangeUpdate(SoftUpdate):
    def __set__(self, obj, value):
        if not hasattr(obj, 'npars'):
            return
        print(f'Updating {self.private_name[1:]} to {value}')
        pmins = np.nanmin(obj.posterior, axis=0)
        pmaxs = np.nanmax(obj.posterior, axis=0)
        min_max = [(pmin, pmax) for pmin,pmax in zip(pmins, pmaxs)]
        if value is None:
            value = min_max
        if len(value) != obj.npars:
            print(f'Invalid {self.private_name[1:]} value: {value}')
            return
        value = [
            min_max[i] if value[i] is None else value[i]
            for i in range(obj.npars)]
        setattr(obj, self.private_name, value)


class Posterior(object):
    """Classification of posterior plotting tools.

    Milestones
    ----------
    + Get basics working: pair(+hist) plot.
    + Allow tweaks/replotting: set_geom(rect, margin), set_fs, set_ranges()
    - Get basics working: histogram plot.
    - Allow adding posteriors

    Examples
    --------
    >>> import mc3
    >>> import mc3.stats as ms
    >>> from mc3.plots import subplot
    >>> mcmc = np.load('MCMC_HD209458b_sing_0.29-2.0um_MM2017.npz')
    >>> posterior, zchain, zmask = mc3.utils.burn(mcmc)

    >>> idx = np.arange(7, 13)
    >>> post = posterior[:,idx]
    >>> pnames = mcmc['texnames'][idx]
    >>> bestp = mcmc['bestp'][idx]
    >>> self = p = mc3.plots.Posterior(post, pnames, bestp)
    >>> p.plot()
    >>> p.rect = (0.12, 0.12, 0.98, 0.98)
    >>> new_pnames = [
           '$\\log_{10}(X_{\\rm Na})$',  '$\\log_{10}(X_{\\rm K})$',
           '$\\log_{10}(X_{\\rm H2O})$', '$\\log_{10}(X_{\\rm CH4})$',
           '$\\log_{10}(X_{\\rm NH3})$', '$\\log_{10}(X_{\\rm HCN})$']
    >>> p.pnames = new_pnames  # Auto-updates
    >>> p.pnames[0] = '$\\log_{10}(X_{\\rm Na})$' # Does not auto-update
    >>> # Update call:
    >>> p.update(rect=(0.12, 0.12, 0.98, 0.98))
    >>> p.update(rect=(0.1, 0.1, 0.98, 0.98), quantile=None)
    >>> p.update(ranges=[(-8, -2) for _ in pnames])
    """
    # Soft-update properties:
    pnames = SoftUpdate()
    rect = SoftUpdate()
    margin = SoftUpdate()
    ymargin = SoftUpdate()
    fontsize = SoftUpdate()
    plot_marginal = SoftUpdate()
    figsize = SizeUpdate()

    bestp = BestpUpdate()
    ranges = RangeUpdate()
    thinning = ThinningUpdate()
    theme = ThemeUpdate()
    #fignum='Pairwise posterior',

    def __init__(self, posterior, pnames=None, bestp=None, ranges=None,
            thinning=1, quantile=0.683,
            bins=25, nlevels=20, fontsize=11, linewidth=1.5,
            rect=None, figsize=None,
            margin=0.01, ymargin=None, pdf=None, xpdf=None,
            plot_marginal=True,
            theme='default', orientation='vertical',
            fignum='Pairwise posterior', savefile=None,):

        # TBD: check size(post) matches size of pnames
        self.input_posterior = posterior
        self.thinning = thinning
        nsamples, self.npars = np.shape(posterior)

        self.pair_axes = None
        self.hist_axes = None

        # Defaults:
        if pnames is None:
            pnames = mu.default_parnames(self.npars)

        if rect is None:
            rect = (0.1, 0.1, 0.98, 0.98)

        if figsize is None:
            figsize = (8,8)

        self.fig = None
        self.pnames = pnames
        self.bestp = bestp
        self.ranges = ranges
        self.quantile = quantile
        self.fignum = fignum
        self.bins = bins
        self.nlevels = nlevels
        #self.ms = ms
        self.fontsize = fontsize
        self.linewidth = linewidth
        self.rect = rect
        self.figsize = figsize
        self.savefile = savefile
        self.theme = theme
        self.margin = margin
        self.ymargin = ymargin
        self.orientation = orientation
        self.plot_marginal = plot_marginal

        if pdf is None or xpdf is None:
            self.pdf = [None]*self.npars
            self.xpdf = [None]*self.npars
        else:
            self.pdf = pdf
            self.xpdf = xpdf

    def plot(self, plot_marginal=None, fignum=None):
        """
        Defaults to histogram plus pairwise
        """
        if self.pair_axes is None:
            self.pair_axes = np.tile(None, (self.npars-1, self.npars-1))

        nx = self.npars - int(not self.plot_marginal)
        #plt.close(self.fignum)
        self.fig = plt.figure(self.fignum, self.figsize)
        plt.clf()
        for icol in range(self.npars-1):
            for irow in range(icol, self.npars-1):
                h = nx*irow + icol + 1 + self.npars*int(self.plot_marginal)
                self.pair_axes[irow,icol] = subplot(
                    self.rect, self.margin, h, nx, ymargin=self.ymargin)

        self.palette = copy.copy(self.theme['colormap'])
        self.palette.set_under(color='w')
        self.palette.set_bad(color='w')

        absolute_dens = False
        self.hist_xran, self.hist, self.lmax = hist_2D(
            self.posterior, self.ranges, self.bins, self.nlevels)
        _pairwise(
            self.posterior, self.pnames, self.bestp, self.ranges,
            self.pair_axes,
            self.bins, self.nlevels,
            absolute_dens, self.palette,
            self.fontsize,
            self.rect, self.hist_xran, self.hist, self.lmax,
        )
        # The colorbar:
        dx = (self.rect[2]-self.rect[0])*0.05
        dy = (self.rect[3]-self.rect[1])*0.45
        bounds = np.linspace(0, 1.0, self.nlevels)
        self.colorbar = colorbar = mpl.colorbar.ColorbarBase(
            plt.axes([self.rect[2]-dx, self.rect[3]-dy, dx, dy]),
            cmap=self.palette,
            norm=mpl.colors.BoundaryNorm(bounds, self.palette.N),
            spacing='proportional', boundaries=bounds, format='%.1f')
        colorbar.set_label("Posterior density", fontsize=self.fontsize)
        colorbar.ax.yaxis.set_ticks_position('left')
        colorbar.ax.yaxis.set_label_position('left')
        colorbar.ax.tick_params(
            labelsize=self.fontsize-1, direction='in', right=True)
        colorbar.set_ticks(np.linspace(0, 1, 6))
        for c in colorbar.ax.collections:
            c.set_edgecolor("face")

        # Marginal posterior:
        if self.plot_marginal:
            if self.hist_axes is None:
                self.hist_axes = np.tile(None, self.npars)
            for i in range(self.npars):
                h = (self.npars+1)*i + 1
                self.hist_axes[i] = subplot(
                    self.rect, self.margin, h, self.npars, ymargin=self.ymargin)

            self.hist_pnames = [
                '' if i < self.npars-1 else self.pnames[i]
                for i in range(self.npars)]
            self.plot_histogram(
                self.hist_axes, self.hist_pnames, quantile=self.quantile)


    def plot_histogram(self, axes=None, pnames=None, quantile=0.683):
        """Plot the marginal histograms of the posterior distribution"""
        # if figs is None or axes is None: self.fig =
        if quantile is None:
            pass
        elif quantile != self.quantile or self.pdf[0] is None:
            for i in range(self.npars):
                self.pdf[i], self.xpdf[i], hpd_min = ms.cred_region(
                    self.posterior[:,i], quantile, self.pdf[i], self.xpdf[i])
        self.quantile = quantile

        yscale = False

        _histogram(
            self.posterior, pnames, self.bestp, self.ranges,
            self.hist_axes,
            self.bins, self.quantile, self.pdf, self.xpdf,
            self.linewidth, self.fontsize, self.theme, yscale, self.orientation)

    def add():
        """Add another posterior"""
        pass

    def update(self, **kwargs):
        for key, value in kwargs.items():
            # If key in valid keys:
            print(f'{key} = {value}')
            # Else: throw warning
        replot = False
        # if hard-plot update parameter in kwargs.keys():
        # 'thinning' in kwargs.keys():
        # 'bestp' in kwargs.keys():
        # 'ranges' in kwargs.keys():
        # 'linewidth' in kwargs.keys():
        # 'theme' in kwargs.keys()
        # 'quantile'
        # 'bins'
        # 'nlevels'
        replot = True

        if len(kwargs) == 0:
            replot = True

        for key, value in kwargs.items():
            setattr(self, key, value)

        if replot:
            self.plot()


    def plot_fit():
        pass

    def plot_trace():
        pass

