# Copyright (c) 2015-2023 Patricio Cubillos and contributors.
# mc3 is open-source software under the MIT license (see LICENSE).

__all__ = [
    # Functions:
    'rms',
    'trace',
    # To be deprecated:
    'modelfit',
    'histogram',
    'pairwise',
    'subplotter',
]

import os

import numpy as np
import matplotlib.pyplot as plt

from .. import utils as u
from .. import stats as ms


def rms(
        binsz, rms, stderr, rmslo, rmshi, cadence=None, binstep=1,
        timepoints=[], ratio=False, fignum=1300,
        yran=None, xran=None, savefile=None,
    ):
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
        xlabel = 'Bin size'
    else:
        xlabel = 'Bin size (seconds)'

    if yran is None:
        yran = [np.amin(rms-rmslo), np.amax(rms+rmshi)]
        yran[0] = np.amin([yran[0],stderr[-1]])
        if ratio:
            yran = [0, np.amax(rms/stderr) + 1.0]
    if xran is None:
        xran = [cadence, np.amax(binsz*cadence)]

    fs = 14  # Font size
    ylabel = r'$\beta$ = RMS / Gaussian noise' if ratio else 'RMS'

    plt.figure(fignum, (8,6))
    plt.clf()
    ax = plt.subplot(111)
    if ratio:
        ax.errorbar(
            binsz[::binstep]*cadence, (rms/stderr)[::binstep],
            yerr=[(rmslo/stderr)[::binstep], (rmshi/stderr)[::binstep]],
            fmt='k-', ecolor='0.5', capsize=0, label="__nolabel__",
        )
        ax.semilogx(xran, [1,1], "r-", lw=2)
    else:
        # Residuals RMS:
        ax.errorbar(
            binsz[::binstep]*cadence, rms[::binstep],
            yerr=[rmslo[::binstep], rmshi[::binstep]],
            fmt='k-', ecolor='0.5', capsize=0, label='RMS')
        # Gaussian noise projection:
        ax.loglog(
            binsz*cadence, stderr, color='red', ls='-', lw=2.0,
            label='Gaussian noise',
        )
        ax.legend(loc='best')

    for time in timepoints:
        ax.vlines(time, yran[0], yran[1], 'b', 'dashed', lw=2)

    ax.tick_params(
        labelsize=fs-1, direction='in', top=True, right=True, which='both',
    )
    ax.set_ylim(yran)
    ax.set_xlim(xran)
    ax.set_ylabel(ylabel, fontsize=fs)
    ax.set_xlabel(xlabel, fontsize=fs)

    if savefile is not None:
        plt.savefig(savefile)
    return ax


def trace(
        posterior, zchain=None, pnames=None,
        burnin=0, fignum=1000, savefile=None, fmt=".", ms=2.5, fs=10,
        color='xkcd:blue',
    ):
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
    color: string
        A color.

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
        zchain = zchain[good]
        # Sort the posterior by chain:
        zsort = np.lexsort([zchain])
        posterior = posterior[zsort]
        zchain = zchain[zsort]
        # Get location for chains separations:
        xsep = np.where(np.ediff1d(zchain))[0]

    nsamples, npars = np.shape(posterior)
    npanels = 12  # Max number of panels per page
    npages = int(1 + (npars-1)/npanels)

    if pnames is None:
        pnames = u.default_parnames(npars)

    # Make the trace plot:
    axes = []
    ipar = 0
    axis_height = 0.62
    hspace = 0.15
    for page in range(npages):
        fig = plt.figure(fignum+page)
        plt.clf()
        nx = np.clip(npars-ipar, 0, npanels)
        height = axis_height *(nx + (nx-1)*hspace) + 0.65
        fig.set_size_inches(8.0, height)
        bottom = 0.45 / height
        top = 1.0 - 0.20 / height
        plt.subplots_adjust(
            left=0.15, right=0.98, bottom=bottom, top=top, hspace=hspace)
        while ipar < npars:
            ax = plt.subplot(nx, 1, ipar%npanels+1)
            axes.append(ax)
            ax.plot(posterior[:,ipar], fmt, ms=ms, color=color)
            yran = ax.get_ylim()
            if zchain is not None:
                ax.vlines(xsep, yran[0], yran[1], '0.2', lw=0.75, zorder=-10)

            ax.set_ylim(yran)
            ax.locator_params(axis='y', nbins=5, tight=True)
            ax.tick_params(labelsize=fs-1, direction='in', top=True, right=True)
            ax.set_ylabel(pnames[ipar], size=fs, multialignment='center')
            ax.set_xlim(0, nsamples)
            ax.get_xaxis().set_visible(False)
            ipar += 1
            if ipar%npanels == 0:
                break
        ax.set_xlabel('MCMC sample', size=fs)
        ax.get_xaxis().set_visible(True)

        if savefile is not None:
            if npages > 1:
                name, extension = os.path.splitext(savefile)
                fig.savefig(f"{name}_page{page+1:02d}{extension}", dpi=300)
            else:
                fig.savefig(savefile, dpi=300)

    return axes


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Deprecated functions:

def histogram(
        posterior, pnames=None, thinning=1, fignum=1100,
        savefile=None, bestp=None, quantile=None, pdf=None,
        xpdf=None, ranges=None, axes=None, lw=2.0, fs=11,
        nbins=25, theme='blue', yscale=False, orientation='vertical',
        statistics='med_central',
    ):
    """
    Deprecated function. Use the plot_histogram() function of
    mc3.plots.Posterior() instead.
    """
    from .posterior import Posterior
    post = Posterior(
        posterior, pnames=pnames, bestp=bestp, theme=theme,
        quantile=quantile, ranges=ranges,
        statistics=statistics,
    )
    fig = post.plot_histogram(
        savefile=savefile, fignum=fignum, #linewidth=lw, fontsize=fs,
        axes=axes,
    )
    return fig


def pairwise(
        posterior, pnames=None, thinning=1, fignum=1200,
        savefile=None, bestp=None, nbins=25, nlevels=20,
        absolute_dens=False, ranges=None, fs=11, rect=None, margin=0.01,
        quantile=0.683, theme='blue', statistics='med_central',
        linewidth=2.0, plot_marginal=True,
    ):
    """
    Deprecated function. Use the plot() function of
    mc3.plots.Posterior() instead.
    """
    from .posterior import Posterior
    post = Posterior(
        posterior, pnames=pnames, bestp=bestp, theme=theme,
        quantile=quantile, ranges=ranges,
        statistics=statistics,
    )
    fig = post.plot(
        savefile=savefile, fignum=fignum, linewidth=linewidth, fontsize=fs,
        rect=rect, margin=margin,
    )
    return fig



def modelfit(
        data, uncert, indparams, model, nbins=75,
        fignum=1400, savefile=None, fmt="."
    ):
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
    binmodel = ms.bin_array(model, binsize)
    bindata, binuncert = ms.bin_array(data, binsize, uncert)
    fs = 12

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


def subplotter(
        rect, margin, ipan, nx, ny=None, ymargin=None,
    ):
    """
    Deprecated function. Use mc3.plots.subplot() instead.
    """
    from .posterior import subplot
    return subplot(rect, margin, ipan, nx, ny, ymargin)

