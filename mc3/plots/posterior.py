# Copyright (c) 2015-2023 Patricio Cubillos and contributors.
# mc3 is open-source software under the MIT license (see LICENSE).

__all__ = [
    'subplot',
    '_histogram',
    '_pairwise',
    'hist_2D',
    'Marginal',
    'Figure',
    'Posterior',
]

import copy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers
from matplotlib.colors import is_color_like, to_rgba
from matplotlib.lines import Line2D
import scipy.interpolate as si

from .. import stats as ms
from .. import utils as u
from . import colors


def is_open(fig):
    """Check if a figure has been closed."""
    current_figs = [
        manager.canvas.figure
        for manager in _pylab_helpers.Gcf.figs.values()
    ]
    return fig in current_figs


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


def hist_2D(posterior, ranges, nbins):
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
                posterior[:,irow+1],
                posterior[:,icol],
                bins=nbins, range=ran, density=False,
            )
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
        linewidth, theme, orientation, alpha=0.6,
        top_pad=1.05,
        clear=True,
    ):
    """
    Lowest-lever routine to plot marginal posterior distributions.
    """
    nsamples, npars = np.shape(posterior)
    has_credible_interval = (
        hpd_min is not None
        or low_bounds is not None
    )

    hist_kw = {
        'bins': nbins,
        'linewidth': linewidth,
        'orientation': orientation,
        'facecolor': to_rgba(theme.light_color, alpha=alpha),
        'edgecolor': theme.color,
        'histtype': 'stepfilled',
        'density': True,
    }
    if has_credible_interval:
        hist_kw['facecolor'] = 'none'

    for i in range(npars):
        ax = axes[i]
        if clear:
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
        ytop = top_pad * np.amax(vals)
        if ytop > yax.get_view_interval()[1]:
            yax.set_view_interval(0, ytop, ignore=True)
        xax.set_view_interval(*ranges[i], ignore=True)


def _pairwise(
        hist, hist_xran, axes, ranges, estimates,
        palette, nlevels, absolute_dens, lmax,
        linewidth, theme, alpha=0.8, clear=True,
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
            if clear:
                ax.clear()
            extent = (
                hist_xran[icol,0],
                hist_xran[icol,-1],
                hist_xran[irow+1,0],
                hist_xran[irow+1,-1],
            )
            levels = np.zeros(nlevels+1)
            levels[1:] = np.linspace(1.0, lmax[irow,icol], nlevels)
            colors = palette(levels/lmax[irow,icol], alpha=alpha)
            colors[0,3] = 0.0
            colors[1,3] = 0.75*alpha
            cont = ax.contourf(
                hist[irow,icol],
                colors=colors, levels=levels,
                origin='lower', extent=extent,
            )
            edge_color = to_rgba(theme.color, alpha=0.65)
            for c in cont.collections:
                c.set_edgecolor(edge_color)
                c.set_linewidth(0.1)
            cont.collections[0].set_edgecolor((1,1,1,0))
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
    ax = obj.hist_axes[0]
    fig = ax.get_figure()

    for text in obj.stats_texts:
        text.set_visible(False)
    obj.stats_texts = []
    for i in range(npars):
        ax = obj.hist_axes[i]
        if obj.orientation == 'vertical':
            xax, yax = ax.xaxis, ax.yaxis
        else:
            xax, yax = ax.yaxis, ax.xaxis

        ax.tick_params(
            labelsize=obj.fontsize-1, direction='in', left=False, top=True,
        )
        xax.set_label_text(obj.pnames[i], fontsize=obj.fontsize)
        yax.set_ticklabels([])

        if obj.show_texts:
            texts = [rf'{obj.source.tex_estimates[i]}']
            obj.stats_texts += colors.rainbow_text(
                ax, texts, obj.fontsize-0.25, loc='inside',
            )

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
    colorbar.ax.clear()
    boundaries = np.linspace(0.0, 1.0, obj.nlevels)
    norm = mpl.colors.BoundaryNorm(boundaries, obj.nlevels)
    cmap = mpl.colors.ListedColormap(obj.palette(boundaries, alpha=0.8))
    mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    obj.colorbar = mpl.colorbar.Colorbar(
        ax=colorbar.ax,
        mappable=mappable,
        boundaries=boundaries,
        ticks=np.linspace(0.0, 1.0, 6),
        ticklocation='left',
    )
    colorbar.set_label('Posterior density', fontsize=obj.fontsize)
    colorbar.ax.tick_params(labelsize=obj.fontsize-1, direction='in')
    colorbar.ax.minorticks_off()
    for col in colorbar.ax.collections:
        col.set_edgecolor('face')
    colorbar.ax.set_visible(obj.show_colorbar)

    # Histogram:
    for text in obj.stats_texts:
        text.set_visible(False)
    obj.stats_texts = []
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
            continue
        elif i < npars-1:
            stats_text = rf'{obj.pnames[i]} = {obj.source.tex_estimates[i]}'
        else:
            stats_text = rf'{obj.source.tex_estimates[i]}'
        texts = [stats_text]
        obj.stats_texts += colors.rainbow_text(ax, texts, obj.fontsize)


class SoftUpdate:
    """ https://docs.python.org/3/howto/descriptor.html """
    def __set_name__(self, obj, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        value = getattr(obj, self.private_name)
        return value

    def __set__(self, obj, value):
        var_name = self.private_name[1:]
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
        setattr(obj, self.private_name, tuple(value))
        if obj.fig is not None:
            obj.fig.set_size_inches(*list(value))


class ThemeUpdate(SoftUpdate):
    def __set__(self, obj, value):
        var_name = self.private_name[1:]
        # TBD: add checks
        if isinstance(value, colors.Theme):
            pass
        elif isinstance(value, str) and value in colors.THEMES:
            value = colors.THEMES[value]
        elif is_color_like(value):
            value = colors.Theme(value)
        setattr(obj, self.private_name, value)
        setattr(obj.source, var_name, value)


class BestpUpdate(SoftUpdate):
    def __set__(self, obj, value):
        var_name = self.private_name[1:]
        if not hasattr(obj, 'npars'):
            return
        if value is None:
            value = [None for _ in range(obj.npars)]
        if len(value) != obj.npars:
            self.raise_array_size_error(obj, value)
        setattr(obj, self.private_name, value)
        setattr(obj.source, var_name, value)


class StatsUpdate(SoftUpdate):
    def __set__(self, obj, value):
        var_name = self.private_name[1:]
        setattr(obj, self.private_name, value)
        setattr(obj.source, var_name, value)


class QuantileUpdate(SoftUpdate):
    def __set__(self, obj, value):
        var_name = self.private_name[1:]
        setattr(obj, self.private_name, value)
        setattr(obj.source, var_name, value)


class RangeUpdate(SoftUpdate):
    def __set__(self, obj, value):
        var_name = self.private_name[1:]
        if not hasattr(obj, 'npars'):
            return
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
    """A mostly-interactive marginal posterior plotting object."""
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
            nx=None, ny=None,
            statistics='med_central', quantile=0.683,
            bins=25, fontsize=11, linewidth=1.5,
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

        if nx is None or ny is None:
            # Default layout:
            npars = self.npars
            if npars < 6:  # Single row, N columns
                nx = npars
            elif npars < 13:  # Two rows, up to 6 columns
                nx = (npars+1) // 2
            elif npars < 25:  # Six columns, up to 4 rows
                nx = 6
            elif npars < 56:  # 7 columns, up to 8 rows
                nx = 7
            else:
                nx = 8  # Stick with 8 columns from now on
            ny = 1 + (npars-1) // nx
        self.nx = nx
        self.ny = ny
        # Default layout sizes:
        dx0 = 0.4
        self.figsize = size = [
            dx0 + 1.45*self.nx,
            1.75*self.ny + 0.1
        ]
        self.margin = 0.04 / self.nx
        self.ymargin = 0.275 / self.ny
        self.rect = [
            dx0/size[0], self.ymargin, 1.0 - 0.2/size[0], 1.0 - 0.1/size[1],
        ]

        self.statistics = statistics
        self.quantile = quantile
        self.bins = bins
        self.fontsize = fontsize
        self.linewidth = linewidth
        self.orientation = 'vertical'
        self.show_texts = show_texts
        self.show_estimates = show_estimates
        self.stats_texts = []

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
        # Create new figure unless explicitly point to existing axes:
        if axes is not None:
            self.hist_axes = axes
            self.fig = axes[0].get_figure()
            self.auto_axes = False
        else:
            self.fig = plt.figure(fignum)
            self.fig.set_size_inches(self.figsize)
            self.fig.clear()
            self.auto_axes = True  # False when user inputs custom axes
        self.fignum = self.fig.number
        self.figsize = self.fig.get_size_inches()

        if axes is None:
            self.hist_axes = np.tile(None, self.npars)
            for i in range(self.npars):
                self.hist_axes[i] = subplot(
                    self.rect, self.margin, i+1, self.nx, self.ny, self.ymargin,
                )

        if '_like' in self.statistics:
            hpd_min = self.source.hpd_min
        else:
            hpd_min = None

        estimates = self.source.estimates
        if not self.show_estimates:
            estimates = [None for _ in estimates]

        _histogram(
            self.posterior, estimates, self.ranges,
            self.hist_axes, self.bins,
            self.source.pdf, self.source.xpdf,
            hpd_min, self.source.low_bounds, self.source.high_bounds,
            self.linewidth, self.theme,
            self.orientation,
            top_pad=1.25,
        )
        _plot_marginal(self)

        if savefile is not None:
            self.fig.savefig(savefile, dpi=300)


class Figure(Marginal):
    """A mostly-interactive pair-wise posterior plotting object."""
    # Soft-update properties:
    plot_marginal = SoftUpdate()
    show_colorbar = SoftUpdate()

    def __init__(
            self, source, posterior, pnames, bestp, ranges, theme,
            plot_marginal=True,
            figsize=None, rect=None, margin=None, ymargin=None,
            statistics='med_central', quantile=0.683,
            bins=25, nlevels=6, fontsize=None, linewidth=None,
            show_texts=True, show_estimates=True,
            show_colorbar=True,
            fignum=None,
        ):
        self.source = source
        self.fig = None
        self.pair_axes = None
        self.hist_axes = None
        self.posterior = posterior
        self.stats_texts = []
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
        self.fignum = fignum
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
            self.plot_marginal, figure=self.fig,
        )

    def plot(
            self, plot_marginal=True, figure=None,
            savefile=None,
        ):
        """Pairwise plus histogram plot."""
        # Can't plot if there are no pairs:
        if self.npars == 1:
            return
        # Create new figure unless explicitly point to an existing one:
        if figure is not None and plt.fignum_exists(figure.number):
            self.fig = figure
        else:
            self.fig = plt.figure(self.fignum)
            self.fig.clear()
            self.fig.set_size_inches(self.figsize)
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
            self.posterior, self.ranges, self.bins,
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
            _histogram(
                self.posterior, estimates, self.ranges,
                self.hist_axes, self.bins,
                self.source.pdf, self.source.xpdf,
                hpd_min,
                self.source.low_bounds, self.source.high_bounds,
                self.linewidth, self.theme,
                self.orientation,
            )
        _plot_pairwise(self)

        if savefile is not None:
            self.fig.savefig(savefile, dpi=300)

    def overplot(self, posts, labels=None, nlevels=4, alpha=0.4):
        """
        Overplot additional posteriors in the same figure.

        This method is still work in progress!
        Note that a call to self.update() or even soft updates
        will remove all/some of the overplot data. In such case
        the user would need to make a new call to self.overplot().
        It is also recommended to set show_estimates=False to
        prevent over-crowding the figures.

        Parameters
        ----------
        posts: 1D iterable of Posterior objects
            Currently there are no checks that these new posteriors
            have the same parameters (nor same statistics) as self.
            The user needs to make sure they are all compartible.
        labels: 1D iterable of strings
            Labels for each posterior.  Note that if provided, the
            length of labels has to be one more than posts, because
            it also contains the label for self.
        """
        all_posts = [self.source] + list(posts)
        for post in posts:
            post.hist_xran, post.hist, post.lmax = hist_2D(
                post.posterior, self.ranges, self.bins,
            )
            absolute_dens = False
            estimates = post.estimates
            if not self.show_estimates:
                estimates = [None for _ in estimates]

            # Pair-wise plots:
            _pairwise(
                post.hist, post.hist_xran,
                self.pair_axes, self.ranges, estimates,
                post.theme.colormap,
                nlevels,
                absolute_dens,
                post.lmax,
                self.linewidth,
                post.theme,
                alpha=alpha,
                clear=False,
            )
        # Posterior labels:
        ax = self.hist_axes[0] if self.plot_marginal else self.pair_axes[0,0]
        if labels is not None:
            ret_handles = [
                Line2D([], [], color=post.theme.color, label=label)
                for post,label in zip(reversed(all_posts),reversed(labels))
            ]
            if self.plot_marginal and self.show_texts:
                loc = (2.1, 1.0)
            else:
                loc = (1.1, 0.1)
            leg = ax.legend(
                handles=ret_handles, loc=loc,
                fontsize=self.fontsize,
                labelspacing=0.25,
            )
            self.legend = leg

        # Histogram plots:
        if not self.plot_marginal:
            return
        for post in posts:
            estimates = post.estimates
            if not self.show_estimates:
                estimates = [None for _ in estimates]
            hpd_min = None
            if '_like' in self.statistics:
                hpd_min = self.source.hpd_min

            _histogram(
                post.posterior, estimates, self.ranges, self.hist_axes,
                self.bins, post.pdf, post.xpdf,
                hpd_min, post.low_bounds, post.high_bounds,
                self.linewidth, post.theme, post.orientation, alpha=0.5,
                clear=False,
            )
        if not self.show_texts:
            return
        text_cols = [post.theme.color for post in all_posts]
        for text in self.stats_texts:
            text.set_visible(False)
        self.stats_texts = []

        for j in range(self.npars):
            stats_texts = [
                rf'{self.pnames[j]} = {post.tex_estimates[j]}'
                for post in all_posts
            ]
            if j == self.npars-1:
                stats_texts = [text[text.index('=')+2:] for text in stats_texts]
            self.stats_texts += colors.rainbow_text(
                self.hist_axes[j], stats_texts, self.fontsize, text_cols,
            )


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
        setattr(obj, priv_name, value)
        for i in reversed(range(len(obj.figures))):
            fig = obj.figures[i]
            if not is_open(fig.fig):
                obj.figures.pop(i)
            else:
                setattr(fig, var_name, value)


class ShareTheme(ShareUpdate):
    def __set__(self, obj, value):
        priv_name = self.private_name
        var_name = self.private_name[1:]
        if isinstance(value, colors.Theme):
            pass
        elif isinstance(value, str) and value in colors.THEMES:
            value = colors.THEMES[value]
        elif is_color_like(value):
            value = colors.Theme(value)
        if hasattr(obj, priv_name) and value == getattr(obj, priv_name):
            return
        setattr(obj, priv_name, value)
        for i in reversed(range(len(obj.figures))):
            fig = obj.figures[i]
            if not is_open(fig.fig):
                obj.figures.pop(i)
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
    >>> pnames = mcmc['texnames']
    >>> bestp = mcmc['bestp']

    >>> p = mc3.plots.Posterior(posterior, pnames, bestp)
    >>> f1 = p.plot(savefile=f'pairwise_{6:02d}pars.png')
    >>> f2 = p.plot_histogram(savefile=f'histogram_{6:02d}pars.png')
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
            self, plot_marginal=True,
            fignum=None, figure=None,
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
            fignum=fignum,
            show_texts=show_texts,
            show_estimates=show_estimates,
            show_colorbar=show_colorbar,
            # bins=25, nlevels=20,
        )
        self.figures.append(fig)
        fig.plot(figure=figure, savefile=savefile)
        return fig


    def plot_histogram(
            self, fignum=None, axes=None, quantile=None,
            nx=None, ny=None,
            savefile=None,
            show_texts=None, show_estimates=None,
        ):
        """
        Plot histogram of marginal posteriors.
        """
        if show_estimates is None:
            show_estimates = self.show_estimates
        if show_texts is None:
            show_texts = self.show_texts

        fig = Marginal(
            self,
            self.posterior, self.pnames, self.bestp,
            self.ranges, self.theme,
            statistics=self.statistics,
            nx=nx, ny=ny,
            show_texts=show_texts,
            show_estimates=show_estimates,
        )
        self.figures.append(fig)
        fig.plot(savefile=savefile, fignum=fignum, axes=axes)
        return fig


    def add():
        """TBD: Do not call this method."""
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
