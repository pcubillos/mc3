# Copyright (c) 2015-2023 Patricio Cubillos and contributors.
# mc3 is open-source software under the MIT license (see LICENSE).

__all__ = [
    # Functions:
    'rms',
    'modelfit',
    'subplot',
    '_histogram',
    '_pairwise',
    'hist_2D',
    # Objects:
    'Posterior',
    'Marginal',
    'Figure',
    # To be deprecated:
    'trace',
    'histogram',
    'pairwise',
    'subplotter',
]

import copy
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib import _pylab_helpers
from matplotlib.colors import is_color_like
import scipy.interpolate as si

from .. import stats as ms
from .. import utils as u
from . import colors


tick_scale = 1/50.0


def is_open(fig):
    """Check if a figure has not been closed."""
    current_figs = [
        manager.canvas.figure
        for manager in _pylab_helpers.Gcf.figs.values()
    ]
    return fig in current_figs


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


def modelfit(
        data, uncert, indparams, model, nbins=75,
        fignum=1400, savefile=None, fmt="."
    ):
    pass


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


def trace(
        posterior, zchain=None, pnames=None,
        burnin=0, fignum=1000, savefile=None, fmt=".", ms=2.5, fs=10,
        color='xkcd:blue',
    ):
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
    npanels = 16  # Max number of panels per page
    npages = int(1 + (npars-1)/npanels)

    if pnames is None:
        pnames = u.default_parnames(npars)

    # Make the trace plot:
    axes = []
    ipar = 0
    axis_height = 0.554
    hspace = 0.15
    for page in range(npages):
        fig = plt.figure(1000)
        nx = np.clip(npars-ipar, 0, npanels)
        height = nx*axis_height + (nx-1)*hspace*axis_height + 0.88
        fig.set_size_inches(8.5, height)
        bottom = 0.55 / height
        top = 1.0 - 0.33 / height
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
                sf = os.path.splitext(savefile)
                fig.savefig(f"{sf[0]}_page{page+1:02d}{sf[1]}", dpi=300)
            else:
                fig.savefig(savefile, dpi=300)

    return axes


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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


def _histogram(
        posterior, estimates, ranges, axes,
        nbins, pdf, xpdf, hpd_min, low_bounds, high_bounds,
        linewidth, theme, yscale, orientation, alpha=0.6,
    ):
    """
    Lowest-lever routine to plot marginal posterior distributions.
    """
    nsamples, npars = np.shape(posterior)
    has_credible_interval = (
        hpd_min is not None
        or low_bounds is not None
    )

    # Put all other keywords here?
    hist_kw = {
        'bins': nbins,
        'linewidth': linewidth,
        'orientation': orientation,
        'facecolor': to_rgba(theme.light_color, alpha=alpha),
        'edgecolor': theme.color,
        'histtype': 'stepfilled',
        'density': not yscale,
    }
    if has_credible_interval:
        hist_kw['facecolor'] = 'none'

    maxylim = 0
    for i in range(npars):
        ax = axes[i]
        ax.clear()
        if orientation == 'vertical':
            xax, yax = ax.xaxis, ax.yaxis
            fill_between = ax.fill_between
            axline = ax.axvline
        else:
            xax, yax = ax.yaxis, ax.xaxis
            fill_between = ax.fill_betweenx
            axline = ax.axhline

        vals, bins, h = ax.hist(posterior[:,i], range=ranges[i], **hist_kw)
        # Plot the credible intervals as shaded areas:
        if has_credible_interval:
            vals = np.r_[0, vals, 0]
            bins = np.r_[bins[0] - (bins[1]-bins[0]), bins]
            f = si.interp1d(bins+0.5*(bins[1]-bins[0]), vals, kind='nearest')
            xran = (xpdf[i]>ranges[i][0]) & (xpdf[i]<ranges[i][1])

            if hpd_min is not None:
                x_shade = pdf[i][xran] >= hpd_min[i]
            elif low_bounds is not None:
                x_shade = (
                    (xpdf[i][xran] >= low_bounds[i]) &
                    (xpdf[i][xran] <= high_bounds[i])
                )
            fill_between(
                xpdf[i][xran], 0.0, f(xpdf[i][xran]),
                where=x_shade,
                facecolor=theme.light_color,
                edgecolor='none',
                interpolate=False,
                alpha=alpha,
            )

        if estimates[i] is not None:
            axline(
                estimates[i],
                dashes=(9,2),
                lw=linewidth,
                color=theme.dark_color,
            )
        maxylim = np.amax((maxylim, yax.get_view_interval()[1]))
        xax.set_view_interval(*ranges[i], ignore=True)

    if yscale:
        for ax in axes:
            yax = ax.yaxis if orientation=='vertical' else ax.xaxis
            yax.set_view_interval(0, maxylim, ignore=True)


def _pairwise(
        hist, hist_xran, axes, ranges, estimates,
        palette, nlevels, absolute_dens, lmax,
        linewidth, theme,
    ):
    """
    Lowest-lever routine to plot pair-wise posterior distributions.
    (Everything happening inside the axes)
    """
    npars = len(ranges)
    # Reset upper boundary to absolute maximum value if requested:
    if absolute_dens:
        lmax[:] = np.amax(lmax)

    for icol in range(npars-1):
        for irow in range(icol, npars-1):
            ax = axes[irow,icol]
            # TBD: Do not clear final product?
            ax.clear()
            # The plot:
            levels = [0] + list(np.linspace(1,lmax[irow,icol], nlevels))
            extent = (
                hist_xran[icol,0],
                hist_xran[icol,-1],
                hist_xran[irow+1,0],
                hist_xran[irow+1,-1],
            )
            cont = ax.contourf(
                hist[irow,icol], cmap=palette, vmin=1, origin='lower',
                levels=levels,
                extent=extent,
            )
            for c in cont.collections:
                c.set_edgecolor("face")
            if estimates[icol] is not None:
                ax.axvline(
                    estimates[icol],
                    dashes=(9,2), lw=linewidth, color=theme.dark_color,
                )
            if estimates[irow+1] is not None:
                ax.axhline(
                    estimates[irow+1],
                    dashes=(9,2), lw=linewidth, color=theme.dark_color,
                )
            if ranges[icol] is not None:
                ax.set_xlim(ranges[icol])
            if ranges[irow] is not None:
                ax.set_ylim(ranges[irow+1])


def _plot_marginal(obj):
    """Re-draw everything except the data inside the axes."""
    npars = obj.npars

    # Estimate size of axes (to later set the length of the ticks)
    ax = obj.hist_axes[0]
    fig = ax.get_figure()

    for i in range(npars):
        ax = obj.hist_axes[i]
        if obj.orientation == 'vertical':
            xax, yax = ax.xaxis, ax.yaxis
            #plt.setp(xax.get_majorticklabels(), rotation=90)
        else:
            xax, yax = ax.yaxis, ax.xaxis

        ax.tick_params(
            labelsize=obj.fontsize-1, direction='in', left=False, top=True,
        )
        xax.set_label_text(obj.pnames[i], fontsize=obj.fontsize)
        yax.set_ticklabels([])

        if not obj.show_texts:
            stats_text = None
        else:
            stats_text = rf'{obj.source.tex_estimates[i]}'
        ax.set_title(stats_text, fontsize=obj.fontsize, loc='left')

        if not obj.auto_axes:
            continue
        ax_position = subplot(
            obj.rect, obj.margin, i+1, obj.nx, obj.ny, obj.ymargin, dry=True,
        )
        ax.set_position(ax_position)
        if i == 0:
            pt_to_pix = fig.canvas.get_renderer().points_to_pixels(72.0)
            axes_size_pix = np.amin(ax.get_window_extent().size)
            axes_size_pt = axes_size_pix / pt_to_pix * 72.0
            tick_size = np.amin([3.5, axes_size_pt/15.0])
        ax.tick_params(length=tick_size)

        if i%obj.nx == 0:
            yax.set_label_text('Posterior', fontsize=obj.fontsize)


def _plot_pairwise(obj):
    """Re-draw everything except the data inside the axes."""
    npars = obj.npars

    # Estimate size of axes (to later set the length of the ticks)
    ax = obj.pair_axes[0,0]
    fig = ax.get_figure()

    nx = npars + int(obj.plot_marginal) - 1
    for icol in range(npars-1):
        for irow in range(icol, npars-1):
            ax = obj.pair_axes[irow,icol]
            h = nx*irow + icol + 1 + npars*int(obj.plot_marginal)
            ax_position = subplot(
                obj.rect, obj.margin, h, nx, nx, obj.ymargin, dry=True,
            )
            ax.set_position(ax_position)
            if icol==0 and irow==0:
                pt_to_pix = fig.canvas.get_renderer().points_to_pixels(72.0)
                axes_size_pix = np.amin(ax.get_window_extent().size)
                axes_size_pt = axes_size_pix / pt_to_pix * 72.0
                tick_size = np.amin([3.5, axes_size_pt/15.0])
            # Labels:
            ax.tick_params(
                labelsize=obj.fontsize-1,
                length=tick_size,
                direction='in',
            )
            if icol == 0:
                ax.set_ylabel(obj.pnames[irow+1], size=obj.fontsize)
            else:
                ax.set_yticklabels([])
            if irow == npars-2:
                ax.set_xlabel(obj.pnames[icol], size=obj.fontsize)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
            else:
                ax.set_xticklabels([])

    # Re-draw the colorbar:
    colorbar = obj.colorbar
    dx = (obj.rect[2]-obj.rect[0])*0.03
    dy = (obj.rect[3]-obj.rect[1])*0.45
    colorbar.ax.set_position([obj.rect[2]-dx, obj.rect[3]-dy, dx, dy])
    boundaries = np.linspace(0.0, 1.0, obj.nlevels)
    mappable = mpl.cm.ScalarMappable(
        norm=mpl.colors.BoundaryNorm(boundaries, obj.palette.N),
        cmap=obj.palette,
    )
    obj.colorbar = mpl.colorbar.Colorbar(
        ax=colorbar.ax,
        mappable=mappable,
        boundaries=boundaries,
        ticks=np.linspace(0.0, 1.0, 6),
        ticklocation='left',
    )
    colorbar.set_label('Posterior density', fontsize=obj.fontsize)
    colorbar.ax.tick_params(labelsize=obj.fontsize-1, direction='in')
    for col in colorbar.ax.collections:
        col.set_edgecolor('face')
    colorbar.ax.set_visible(obj.show_colorbar)

    # Histogram:
    nx = npars
    for i in range(npars):
        ax = obj.hist_axes[i]
        ax.set_visible(obj.plot_marginal)
        if not obj.plot_marginal:
            continue
        if obj.orientation == 'vertical':
            xax, yax = ax.xaxis, ax.yaxis
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        else:
            xax, yax = ax.yaxis, ax.xaxis

        h = (npars+1)*i + 1
        ax_position = subplot(
            obj.rect, obj.margin, h, nx, nx, obj.ymargin, dry=True)
        ax.set_position(ax_position)

        ax.tick_params(
            labelsize=obj.fontsize-1,
            length=tick_size,
            direction='in', left=False, top=True,
        )
        if i == npars-1:
            xax.set_label_text(obj.pnames[i], fontsize=obj.fontsize)
        else:
            xax.set_label_text('', fontsize=obj.fontsize)
            xax.set_ticklabels([])
        yax.set_ticklabels([])

        if not obj.show_texts:
            stats_text = None
        elif i < npars-1:
            stats_text = rf'{obj.pnames[i]} = {obj.source.tex_estimates[i]}'
        else:
            stats_text = rf'{obj.source.tex_estimates[i]}'
        ax.set_title(
            stats_text,
            fontsize=obj.fontsize,
            loc='left',
        )


class SoftUpdate:
    """ https://docs.python.org/3/howto/descriptor.html """
    def __set_name__(self, obj, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        value = getattr(obj, self.private_name)
        return value

    def __set__(self, obj, value):
        # TBD: Delete print when done:
        var_name = self.private_name[1:]
        #print(f'Updating {var_name} to {value}')
        setattr(obj, self.private_name, value)
        setattr(obj.source, var_name, value)

        # TBD: Inspect this clause
        if hasattr(obj, 'pair_axes'):
            if obj.pair_axes is not None:
                _plot_pairwise(obj)
                plt.draw()
        else:
            if obj.hist_axes is not None:
                _plot_marginal(obj)
                plt.draw()

    def raise_array_size_error(self, obj, value):
        raise ValueError(
            f"Invalid {self.private_name[1:]} input. Array size "
            f"({len(value)}) does not match number of parameters "
            f"({obj.npars})"
        )


class SizeUpdate(SoftUpdate):
    def __set__(self, obj, value):
        #print(f'Updating {self.private_name[1:]} to {value}')
        setattr(obj, self.private_name, tuple(value))
        if obj.fig is not None:
            obj.fig.set_size_inches(*list(value))


class ThemeUpdate(SoftUpdate):
    def __set__(self, obj, value):
        var_name = self.private_name[1:]
        #print(f'Updating {var_name} to {value}')
        # TBD: add checks
        if isinstance(value, str) and value in colors.THEMES:
            value = colors.THEMES[value]
        elif isinstance(value, str) and is_color_like(value):
            value = colors.Theme(value)
        setattr(obj, self.private_name, value)
        setattr(obj.source, var_name, value)


class BestpUpdate(SoftUpdate):
    def __set__(self, obj, value):
        var_name = self.private_name[1:]
        if not hasattr(obj, 'npars'):
            return
        #print(f'Updating {var_name} to {value}')
        if value is None:
            value = [None for _ in range(obj.npars)]
        if len(value) != obj.npars:
            self.raise_array_size_error(obj, value)
        setattr(obj, self.private_name, value)
        setattr(obj.source, var_name, value)


class StatsUpdate(SoftUpdate):
    def __set__(self, obj, value):
        var_name = self.private_name[1:]
        #print(f'Updating {var_name} to {value}')
        setattr(obj, self.private_name, value)
        setattr(obj.source, var_name, value)


class QuantileUpdate(SoftUpdate):
    def __set__(self, obj, value):
        var_name = self.private_name[1:]
        #print(f'Updating {var_name} to {value}')
        setattr(obj, self.private_name, value)
        setattr(obj.source, var_name, value)


class RangeUpdate(SoftUpdate):
    def __set__(self, obj, value):
        var_name = self.private_name[1:]
        if not hasattr(obj, 'npars'):
            return
        #print(f'Updating {var_name} to {value}')
        pmins = np.nanmin(obj.posterior, axis=0)
        pmaxs = np.nanmax(obj.posterior, axis=0)
        # Defaults:
        min_max = [(pmin, pmax) for pmin,pmax in zip(pmins, pmaxs)]
        if value is None:
            value = min_max
        if len(value) != obj.npars:
            self.raise_array_size_error(obj, value)
        for i in range(obj.npars):
            if value[i] is None:
                value[i] = min_max[i]
        setattr(obj, self.private_name, value)
        setattr(obj.source, var_name, value)


class Marginal(object):
    # Soft-update properties:
    pnames = SoftUpdate()
    rect = SoftUpdate()
    margin = SoftUpdate()
    ymargin = SoftUpdate()
    fontsize = SoftUpdate()
    figsize = SizeUpdate()
    show_texts = SoftUpdate()
    show_estimates = SoftUpdate()

    # Properties that require re-drawing:
    bestp = BestpUpdate()
    ranges = RangeUpdate()
    theme = ThemeUpdate()
    quantile = QuantileUpdate()
    statistics = StatsUpdate()

    def __init__(
            self, source, posterior, pnames, bestp, ranges, theme,
            figsize=None, rect=None, margin=0.005, ymargin=None,
            statistics='med_central', quantile=0.683,
            bins=25, nlevels=20, fontsize=11, linewidth=1.5,
            axes=None,
            show_texts=True, show_estimates=True,
        ):
        self.source = source
        self.fig = None
        self.hist_axes = None
        self.posterior = posterior
        nsamples, self.npars = np.shape(posterior)

        self.pnames = pnames
        self.bestp = bestp
        self.ranges = ranges
        self.theme = theme
        if rect is None:
            rect = [0.1, 0.1, 0.96, 0.96]
        self.rect = rect
        if figsize is None:
            figsize = (8,8)
        self.figsize = figsize
        self.margin = margin
        self.ymargin = ymargin
        self.statistics = statistics
        self.quantile = quantile
        self.bins = bins
        self.nlevels = nlevels
        self.fontsize = fontsize
        self.linewidth = linewidth
        self.orientation = 'vertical'
        self.show_texts = show_texts
        self.show_estimates = show_estimates

    def update(self):
        # TBD: Need to erase previous axes
        self.plot(
            fignum=self.fignum,
        )

    def plot(
            self, fignum=None, axes=None, quantile=None,
            savefile=None,
        ):
        """Marginal histogram plot."""
        npars = self.npars
        # Default layout:
        if npars < 6:  # Single row, N columns
            nx = npars
        elif npars < 13:  # Two rows, up to 6 columns
            nx = (npars+1) // 2
        elif npars < 25:  # Six columns, up to 4 rows
            nx = 6
        else:  # Stick with 4 rows,
            nx = 1 + (npars-1) // 4
        ny = 1 + (npars-1) // nx

        # Default layout sizes:
        dx0 = 0.4
        size = dx0 + 1.45*nx, 2.0*ny
        self.ymargin = 0.3 / ny
        self.rect[0] = dx0/size[0]
        self.rect[1] = self.ymargin

        # Create new figure unless explicitly point to an existing one:
        self.auto_axes = True  # False when user inputs custom axes
        if axes is not None:
            self.hist_axes = axes
            self.fig = axes[0].get_figure()
            self.auto_axes = False
        elif fignum is not None and plt.fignum_exists(fignum):
            self.fig = plt.figure(fignum)
        else:
            self.fig = plt.figure(fignum)
            self.fig.set_size_inches(*list(size))
        self.fignum = self.fig.number
        self.figsize = self.fig.get_size_inches()

        if axes is None:
            self.nx = nx
            self.ny = ny
            self.hist_axes = np.tile(None, npars)
            for i in range(npars):
                self.hist_axes[i] = subplot(
                    self.rect, self.margin, i+1, nx, ny, self.ymargin,
                )


        if '_like' in self.statistics:
            hpd_min = self.source.hpd_min
        else:
            hpd_min = None

        estimates = self.source.estimates
        if not self.show_estimates:
            estimates = [None for _ in estimates]

        yscale = False
        _histogram(
            self.posterior, estimates, self.ranges,
            self.hist_axes, self.bins,
            self.source.pdf, self.source.xpdf,
            hpd_min, self.source.low_bounds, self.source.high_bounds,
            self.linewidth, self.theme,
            yscale, self.orientation,
        )
        _plot_marginal(self)

        if savefile is not None:
            self.fig.savefig(savefile, dpi=300)


class Figure(Marginal):
    # Soft-update properties:
    plot_marginal = SoftUpdate()
    show_colorbar = SoftUpdate()

    def __init__(
            self, source, posterior, pnames, bestp, ranges, theme,
            plot_marginal=True,
            figsize=None, rect=None, margin=None, ymargin=None,
            statistics='med_central', quantile=0.683,
            bins=25, nlevels=20, fontsize=None, linewidth=None,
            show_texts=True, show_estimates=True,
            show_colorbar=True,
        ):
        self.source = source
        self.fig = None
        self.pair_axes = None
        self.hist_axes = None
        self.posterior = posterior
        nsamples, self.npars = np.shape(posterior)

        if fontsize is None:
            # fs(5)=11.0, fs(20)=6.0
            fontsize = np.clip((38.0 -self.npars)/3.0, 6.0, 11.0)
        if linewidth is None:
            # lw(5)=1.5, lw(10)=1.0, lw(20)=0.7
            if self.npars <= 10:
                linewidth = 2.0 - 0.1*self.npars
            else:
                linewidth = 1.3 - 0.03*self.npars
            linewidth = np.clip(linewidth, 0.7, 1.5)
        if margin is None:
            # m(2)=0.01, m(10)=0.005, m(20)=0.003
            if self.npars <= 10:
                margin = 0.01125 - 0.000625*self.npars
            else:
                margin = 0.007 - 0.0002*self.npars
            margin = np.clip(margin, 0.0025, 0.01)

        if figsize is None:
            figsize = (8,8)

        self.pnames = pnames
        self.bestp = bestp
        self.ranges = ranges
        self.theme = theme
        if rect is None:
            rect = [0.1, 0.1, 0.96, 0.96]
        self.rect = rect
        self.figsize = figsize
        self.plot_marginal = plot_marginal
        self.margin = margin
        self.ymargin = ymargin
        self.statistics = statistics
        self.quantile = quantile
        self.bins = bins
        self.nlevels = nlevels
        self.fontsize = fontsize
        self.linewidth = linewidth
        self.orientation = 'vertical'
        self.show_texts = show_texts
        self.show_estimates = show_estimates
        self.show_colorbar = show_colorbar


    def update(self):
        self.plot(
            self.plot_marginal, fignum=self.fignum,
        )

    def plot(
            self, plot_marginal=True, fignum=None,
            savefile=None,
        ):
        """Pairwise plus histogram plot."""
        # Create new figure unless explicitly point to an existing one:
        if fignum is not None and plt.fignum_exists(fignum):
            self.fig = plt.figure(fignum)
        else:
            self.fig = plt.figure(fignum, self.figsize)
        self.fignum = self.fig.number

        # Define the axes:
        npars = self.npars
        nx = npars + int(self.plot_marginal) - 1

        pair_axes_do_not_exist = (
            self.pair_axes is None or
            self.pair_axes[0,0] not in self.fig.axes
        )
        if pair_axes_do_not_exist:
            self.pair_axes = np.tile(None, (npars-1, npars-1))
            for icol in range(npars-1):
                for irow in range(icol, npars-1):
                    h = nx*irow + icol + 1 + npars*int(self.plot_marginal)
                    self.pair_axes[irow,icol] = subplot(
                        self.rect, self.margin, h, nx, ymargin=self.ymargin,
                    )

        self.palette = copy.copy(self.theme.colormap)
        self.palette.set_under(color='w')
        self.palette.set_bad(color='w')

        absolute_dens = False
        self.hist_xran, self.hist, self.lmax = hist_2D(
            self.posterior, self.ranges, self.bins, self.nlevels,
        )

        estimates = self.source.estimates
        if not self.show_estimates:
            estimates = [None for _ in estimates]

        # The pair-wise data:
        _pairwise(
            self.hist, self.hist_xran,
            self.pair_axes, self.ranges, estimates,
            self.palette,
            self.nlevels,
            absolute_dens,
            self.lmax,
            self.linewidth,
            self.theme,
        )

        # The colorbar:
        colorbar_does_not_exist = (
            not hasattr(self, 'colorbar') or
            self.colorbar.ax not in self.fig.axes
        )
        if colorbar_does_not_exist:
            dx = (self.rect[2]-self.rect[0])*0.03
            dy = (self.rect[3]-self.rect[1])*0.45
            cb_ax = plt.axes([self.rect[2]-dx, self.rect[3]-dy, dx, dy])
            mappable = mpl.cm.ScalarMappable()
            self.colorbar = mpl.colorbar.Colorbar(cb_ax, mappable)

        # Marginal posterior:
        hist_axes_do_not_exist = (
            self.hist_axes is None or
            self.hist_axes[0] not in self.fig.axes
        )
        if hist_axes_do_not_exist:
            self.hist_axes = np.tile(None, npars)
            for i in range(npars):
                h = (npars+1)*i + 1
                self.hist_axes[i] = subplot(
                    self.rect, self.margin, h, npars, ymargin=self.ymargin,
                )

        if plot_marginal:
            hpd_min = None
            if '_like' in self.statistics:
                hpd_min = self.source.hpd_min
            yscale = False
            _histogram(
                self.posterior, estimates, self.ranges,
                self.hist_axes, self.bins,
                self.source.pdf, self.source.xpdf,
                hpd_min,
                self.source.low_bounds, self.source.high_bounds,
                self.linewidth, self.theme,
                yscale, self.orientation,
            )
        _plot_pairwise(self)

        if savefile is not None:
            self.fig.savefig(savefile, dpi=300)


class ShareUpdate:
    """ https://docs.python.org/3/howto/descriptor.html """
    def __set_name__(self, obj, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        value = getattr(obj, self.private_name)
        return value

    def __set__(self, obj, value):
        priv_name = self.private_name
        var_name = self.private_name[1:]
        if hasattr(obj, priv_name) and value is getattr(obj, priv_name):
            return
        # TBD: Delete print when done:
        #print(f'Sharing updated value of {var_name} to {value}')
        setattr(obj, priv_name, value)
        #for fig in obj.figures:
        for i in reversed(range(len(obj.figures))):
            fig = obj.figures[i]
            if not is_open(fig.fig):
                obj.figures.pop(i)
                #print(f'pop {i} {fig}')
            else:
                setattr(fig, var_name, value)


class ShareTheme(ShareUpdate):
    def __set__(self, obj, value):
        priv_name = self.private_name
        var_name = self.private_name[1:]
        if isinstance(value, str) and value in colors.THEMES:
            value = colors.THEMES[value]
        elif isinstance(value, str) and is_color_like(value):
            value = colors.Theme(value)
        if hasattr(obj, priv_name) and value == getattr(obj, priv_name):
            return
        #print(f'Sharing updated value of {var_name} to {value}')
        setattr(obj, priv_name, value)
        for i in reversed(range(len(obj.figures))):
            fig = obj.figures[i]
            if not is_open(fig.fig):
                obj.figures.pop(i)
                #print(f'pop {i} {fig}')
            else:
                setattr(fig, var_name, value)


class StatisticsUpdate(ShareUpdate):
    def __set__(self, obj, value):
        var_name = self.private_name[1:]
        priv_name = self.private_name
        if hasattr(obj, priv_name) and value is getattr(obj, priv_name):
            return
        setattr(obj, priv_name, value)
        for i in reversed(range(len(obj.figures))):
            fig = obj.figures[i]
            if not is_open(fig.fig):
                obj.figures.pop(i)
            else:
                setattr(fig, var_name, value)

        has_all_attributes = (
            hasattr(obj, 'bestp') and
            hasattr(obj, 'statistics') and
            hasattr(obj, 'quantile')
        )
        if has_all_attributes:
            #print(f'Now, updating {var_name} to {value}')
            for i in range(obj.npars):
                _, _, obj.hpd_min[i] = ms.cred_region(
                    obj.posterior[:,i],
                    quantile=obj.quantile,
                )
            estimates, low_bounds, high_bounds = ms.marginal_statistics(
                obj.posterior, obj.statistics, obj.quantile,
                pdf=obj.pdf, xpdf=obj.xpdf,
            )
            if obj.statistics.startswith('global_'):
                obj.estimates = obj.bestp
            else:
                obj.estimates = estimates
            obj.low_bounds = low_bounds
            obj.high_bounds = high_bounds

            obj.tex_estimates = u.tex_parameters(
                obj.estimates,
                obj.low_bounds,
                obj.high_bounds,
                significant_digits=2,
            )


class Posterior(object):
    """
    Classification of posterior plotting tools.

    statistics: String
        Statistics to use for parameter estimates and uncertainties:
        global_* use global best-fit (bestp) estimate.
        max_*: Marginal maximum-likelihood (mode) estimate.
        med_*: Marginal median estimate.
        *_like: HPD credible interval.
        *_central: Central quantile interval.

    Examples
    --------
    >>> import mc3

    >>> mcmc = np.load('MCMC_HD209458b_sing_0.29-2.0um_MM2017.npz')
    >>> posterior, zchain, zmask = mc3.utils.burn(mcmc)
    >>> idx = np.arange(7, 13)
    >>> post = posterior[:,idx]
    >>> pnames = mcmc['texnames'][idx]
    >>> bestp = mcmc['bestp'][idx]

    >>> p = mc3.plots.Posterior(post, pnames, bestp)
    >>> f = p.plot(savefile=f'pairwise_{6:02d}pars.png')
    >>> f = p.plot_histogram(savefile=f'histogram_{6:02d}pars.png')

    >>> p2 = mc3.plots.Posterior(posterior, mcmc['texnames'])
    >>> idx20 = np.arange(2) % 16
    >>> post20 = posterior[:,idx20]
    >>> p2 = mc3.plots.Posterior(post20, mcmc['texnames'][idx20])
    >>> f2 = p2.plot()
    >>> plt.savefig(f'pairwise_{2:02d}pars.png', dpi=300)

    >>> for j in [2, 5, 10, 15, 20]:
    >>>     idx20 = np.arange(j) % 16
    >>>     post20 = posterior[:,idx20]
    >>>     p2 = mc3.plots.Posterior(post20, mcmc['texnames'][idx20])
    >>>     f2 = p2.plot()
    >>>     plt.savefig(f'pairwise_{j:02d}pars.png', dpi=300)

    >>> new_pnames = [
           '$\\log_{10}(X_{\\rm Na})$',  '$\\log_{10}(X_{\\rm K})$',
           '$\\log_{10}(X_{\\rm H2O})$', '$\\log_{10}(X_{\\rm CH4})$',
           '$\\log_{10}(X_{\\rm NH3})$', '$\\log_{10}(X_{\\rm HCN})$']
    >>> p.pnames = new_pnames  # Auto-updates
    """
    # Soft-update properties:
    pnames = ShareUpdate()
    ranges = ShareUpdate()
    theme = ShareTheme()

    bestp = StatisticsUpdate()
    statistics = StatisticsUpdate()
    quantile = StatisticsUpdate()

    show_texts = ShareUpdate()
    show_estimates = ShareUpdate()
    show_colorbar = ShareUpdate()

    def __init__(
            self, posterior, pnames=None, bestp=None, ranges=None,
            statistics='med_central', quantile=0.683,
            sample_size=20000,
            theme='blue', orientation='vertical',
            show_texts=True, show_estimates=True,
            show_colorbar=True,
            seed=314159,
        ):
        self.figures = []
        nsamples, self.npars = np.shape(posterior)
        rng = np.random.default_rng(seed)
        if sample_size < nsamples:
            sample = rng.choice(nsamples, sample_size, replace=False)
            sampled_posterior = posterior[sample]
        else:
            sampled_posterior = np.copy(posterior)
        # TBD: enforce posterior as 2D
        self.posterior = sampled_posterior

        # Defaults:
        if pnames is None:
            pnames = [f'p{i:02d}' for i in range(self.npars)]

        self.pnames = pnames
        self.ranges = ranges
        self.theme = theme
        self.orientation = orientation
        self.show_texts = show_texts
        self.show_estimates = show_estimates
        self.show_colorbar = show_colorbar

        self.pdf = [None for _ in range(self.npars)]
        self.xpdf = [None for _ in range(self.npars)]
        self.hpd_min = [None for _ in range(self.npars)]
        for i in range(self.npars):
            pdf, xpdf, hpd = ms.cred_region(
                self.posterior[:,i], quantile=quantile,
            )
            self.pdf[i] = pdf
            self.xpdf[i] = xpdf

        # These will trigger the param estimate calcs in StatisticsUpdate():
        if bestp is None:
            self.bestp = [None for _ in range(self.npars)]
        else:
            self.bestp = bestp
        self.statistics = statistics
        self.quantile = quantile


    def plot(
            self, plot_marginal=True, fignum=None,
            quantile=None,
            linewidth=None, fontsize=None,
            figsize=None, rect=None,
            margin=None, ymargin=None,
            show_texts=None, show_estimates=None,
            show_colorbar=None,
            savefile=None,
        ):
        """
        Plot marginal histograms and pairwise posteriors.
        """
        # Defaults:
        if quantile is None:
            quantile = self.quantile
        if show_estimates is None:
            show_estimates = self.show_estimates
        if show_texts is None:
            show_texts = self.show_texts
        if show_colorbar is None:
            show_colorbar = self.show_colorbar

        fig = Figure(
            self,
            self.posterior, self.pnames, self.bestp,
            self.ranges, self.theme,
            rect=rect,
            margin=margin,
            ymargin=ymargin,
            statistics=self.statistics,
            quantile=quantile,
            plot_marginal=plot_marginal,
            linewidth=linewidth,
            fontsize=fontsize,
            figsize=figsize,
            show_texts=show_texts,
            show_estimates=show_estimates,
            show_colorbar=show_colorbar,
            # bins=25, nlevels=20,
        )
        self.figures.append(fig)
        fig.plot(fignum=fignum, savefile=savefile)
        return fig


    def plot_histogram(
            self, fignum=None, axes=None, quantile=None,
            figsize=None,
            rect=None,
            savefile=None,
            show_texts=False, show_estimates=None,
        ):
        """
        Plot histogram of marginal posteriors.
        """
        if show_estimates is None:
            show_estimates = self.show_estimates

        fig = Marginal(
            self,
            self.posterior, self.pnames, self.bestp,
            self.ranges, self.theme,
            figsize=figsize,
            rect=rect,
            show_texts=show_texts,
            show_estimates=show_estimates,
        )
        self.figures.append(fig)
        fig.plot(savefile=savefile, fignum=fignum, axes=axes)
        return fig


    def add():
        """TBD: Add another posterior"""
        pass

    def update(self, **kwargs):
        for key, value in kwargs.items():
            # If key in valid keys:
            print(f'{key} = {value}')
            # Else: throw warning
        replot = False
        # if hard-replotting, update parameter in kwargs.keys():
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

        #if replot:
        for fig in self.figures:
            fig.update()

    def plot_trace():
        pass


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Deprecated functions:

def histogram(
        posterior, pnames=None, thinning=1, fignum=1100,
        savefile=None, bestp=None, quantile=None, pdf=None,
        xpdf=None, ranges=None, axes=None, lw=2.0, fs=11,
        nbins=25, theme='blue', yscale=False, orientation='vertical'
    ):
    # Deprecated function
    pass


def pairwise(
        posterior, pnames=None, thinning=1, fignum=1200,
        savefile=None, bestp=None, nbins=25, nlevels=20,
        absolute_dens=False, ranges=None, fs=11, rect=None, margin=0.01
    ):
    # Deprecated function
    pass


def subplotter(
        rect, margin, ipan, nx, ny=None, ymargin=None,
    ):
    # TBD: Deprecate warning
    return subplot(rect, margin, ipan, nx, ny, ymargin)

