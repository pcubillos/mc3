# Copyright (c) 2015-2022 Patricio Cubillos and contributors.
# mc3 is open-source software under the MIT license (see LICENSE).

__all__ = [
    'rms',
    'modelfit',
    'themes',
    'subplot',
    '_histogram',
    # Objects:
    'Posterior',
    'Figure',
    # To be deprecated:
    'trace',
    'histogram',
    'pairwise',
    'subplotter',
]

import copy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib import _pylab_helpers
import scipy.interpolate as si

from .. import stats as ms


# Color themes for histogram plots:
themes = {
    'default': {
        'edgecolor': 'blue',
        'facecolor': 'royalblue',
        'color': 'navy',
        'colormap': plt.cm.viridis_r,
    },
    'blue': {
        'edgecolor': 'blue',
        'facecolor': 'royalblue',
        'color': 'navy',
        'colormap': plt.cm.Blues,
    },
    'red': {
        'edgecolor': 'crimson',
        'facecolor': 'orangered',
        'color': 'darkred',
        'colormap': plt.cm.Reds,
    },
    'black': {
        'edgecolor': '0.3',
        'facecolor': '0.3',
        'color': 'black',
        'colormap': plt.cm.Greys,
    },
    'green': {
        'edgecolor': 'forestgreen',
        'facecolor': 'limegreen',
        'color': 'darkgreen',
        'colormap': plt.cm.YlGn,
    },
    'orange': {
        'edgecolor': 'darkorange',
        'facecolor': 'gold',
        'color': 'darkgoldenrod',
        'colormap': plt.cm.YlOrBr,
    },
}


def is_open(fig):
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
   pass


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
        posterior, bestp, ranges, axes,
        nbins, quantile, pdf, xpdf,
        linewidth, theme, yscale, orientation,
    ):
    """
    Lowest-lever routine to plot marginal posterior distributions.
    >>> posterior = self.posterior
    >>> bestp = self.bestp
    >>> ranges = self.ranges
    >>> axes = self.hist_axes
    >>> ticklabels = [ax is axes[-1] for ax in axes]
    >>> yscale = False
    >>> orientation = 'vertical'
    >>> linewidth = self.lw
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
        ax.clear()  # For testing only (?)
        if orientation == 'vertical':
            xax, yax = ax.xaxis, ax.yaxis
            fill_between = ax.fill_between
            axline = ax.axvline
        else:
            xax, yax = ax.yaxis, ax.xaxis
            fill_between = ax.fill_betweenx
            axline = ax.axhline

        vals, bins, h = ax.hist(posterior[:,i], range=ranges[i], **hist_kw)
        # Plot the HPD region as shaded areas:
        if quantile is not None:
            PDF, Xpdf, hpd_min = ms.cred_region(
                posterior[:,i], quantile, pdf[i], xpdf[i])
            vals = np.r_[0, vals, 0]
            bins = np.r_[bins[0] - (bins[1]-bins[0]), bins]
            f = si.interp1d(bins+0.5*(bins[1]-bins[0]), vals, kind='nearest')
            xran = (xpdf[i]>ranges[i][0]) & (xpdf[i]<ranges[i][1])
            fill_between(
                xpdf[i][xran], 0.0, f(xpdf[i][xran]),
                where=pdf[i][xran]>=hpd_min,
                facecolor=theme['facecolor'],
                edgecolor='none',
                interpolate=False,
                alpha=0.6,
            )

        if bestp[i] is not None:
            axline(bestp[i], dashes=(7,4), lw=1.25, color=theme['color'])
        maxylim = np.amax((maxylim, yax.get_view_interval()[1]))
        xax.set_view_interval(*ranges[i], ignore=True)

    if yscale:
        for ax in axes:
            yax = ax.yaxis if orientation=='vertical' else ax.xaxis
            yax.set_view_interval(0, maxylim, ignore=True)


def _pairwise(
        hist, hist_xran, axes, ranges, bestp,
        palette, nlevels, absolute_dens=False, lmax=None,
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
            if bestp[icol] is not None:
                ax.axvline(bestp[icol], dashes=(6,4), color="0.5", lw=1.0)
            if bestp[irow+1] is not None:
                ax.axhline(bestp[irow+1], dashes=(6,4), color="0.5", lw=1.0)
            if ranges[icol] is not None:
                ax.set_xlim(ranges[icol])
            if ranges[irow] is not None:
                ax.set_ylim(ranges[irow+1])


def _plot_pairwise(obj):
    """Re-draw everything except the data inside the axes."""
    npars = obj.npars

    nx = npars + int(obj.plot_marginal) - 1
    for icol in range(npars-1):
        for irow in range(icol, npars-1):
            ax = obj.pair_axes[irow,icol]
            h = nx*irow + icol + 1 + npars*int(obj.plot_marginal)
            ax_position = subplot(
                obj.rect, obj.margin, h, nx, nx, obj.ymargin, dry=True)
            ax.set_position(ax_position)
            # Labels:
            ax.tick_params(labelsize=obj.fontsize-1, direction='in')
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
    bounds = np.linspace(0, 1.0, obj.nlevels)
    obj.colorbar = mpl.colorbar.Colorbar(
        ax=colorbar.ax,
        cmap=obj.palette,
        norm=mpl.colors.BoundaryNorm(bounds, obj.palette.N),
        boundaries=bounds,
        ticks=np.linspace(0.0, 1.0, 6),
        ticklocation='left',
    )
    colorbar.set_label('Posterior density', fontsize=obj.fontsize)
    colorbar.ax.tick_params(
        labelsize=obj.fontsize-1, direction='in', right=True)
    for col in colorbar.ax.collections:
        col.set_edgecolor("face")
    #colorbar.ax.set_visible(False)

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
            labelsize=obj.fontsize-1, direction='in', left=False, top=True)
        if i == npars-1:
            xax.set_label_text(obj.pnames[i], fontsize=obj.fontsize)
        else:
            xax.set_label_text('', fontsize=obj.fontsize)
            xax.set_ticklabels([])
        yax.set_ticklabels([])


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
        print(f'Updating {var_name} to {value}')
        setattr(obj, self.private_name, value)
        setattr(obj.source, var_name, value)

        # TBD: Inspect this clause
        if hasattr(obj, 'pair_axes') and obj.pair_axes is not None:
            _plot_pairwise(obj)

    def raise_array_size_error(self, obj, value):
        raise ValueError(
            f"Invalid {self.private_name[1:]} input. Array size "
            f"({len(value)}) does not match number of parameters "
            f"({obj.npars})"
        )


class SizeUpdate(SoftUpdate):
    def __set__(self, obj, value):
        print(f'Updating {self.private_name[1:]} to {value}')
        setattr(obj, self.private_name, tuple(value))
        if obj.fig is not None:
            obj.fig.set_size_inches(*list(value))


class ThinningUpdate(SoftUpdate):
    def __set__(self, obj, value):
        var_name = self.private_name[1:]
        if not hasattr(obj, 'input_posterior'):
            return
        print(f'Updating {var_name} to {value}')
        setattr(obj, self.private_name, value)
        setattr(obj.source, var_name, value)
        obj.posterior = obj.input_posterior[0::value]


class ThemeUpdate(SoftUpdate):
    def __set__(self, obj, value):
        var_name = self.private_name[1:]
        print(f'Updating {var_name} to {value}')
        # TBD: add checks
        if isinstance(value, str):
            value = themes[value]
        setattr(obj, self.private_name, value)
        setattr(obj.source, var_name, value)


class BestpUpdate(SoftUpdate):
    def __set__(self, obj, value):
        var_name = self.private_name[1:]
        if not hasattr(obj, 'npars'):
            return
        print(f'Updating {var_name} to {value}')
        if value is None:
            value = [None for _ in range(obj.npars)]
        if len(value) != obj.npars:
            self.raise_array_size_error(obj, value)
        setattr(obj, self.private_name, value)
        setattr(obj.source, var_name, value)


class RangeUpdate(SoftUpdate):
    def __set__(self, obj, value):
        var_name = self.private_name[1:]
        if not hasattr(obj, 'npars'):
            return
        print(f'Updating {var_name} to {value}')
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


class Figure(object):
    # Soft-update properties:
    pnames = SoftUpdate()
    rect = SoftUpdate()
    margin = SoftUpdate()
    ymargin = SoftUpdate()
    fontsize = SoftUpdate()
    figsize = SizeUpdate()
    plot_marginal = SoftUpdate()

    # Properties that re-draw:
    thinning = ThinningUpdate()
    bestp = BestpUpdate()
    ranges = RangeUpdate()
    theme = ThemeUpdate()

    def __init__(
            self, source, posterior, pnames, bestp, ranges, theme,
            plot_marginal=True,
            figsize=None, rect=None, margin=0.005, ymargin=None,
            thinning=1, quantile=0.683,
            bins=25, nlevels=20, fontsize=11, linewidth=1.5,
        ):
        self.source = source
        self.fig = None
        self.pair_axes = None
        self.hist_axes = None
        self.input_posterior = posterior
        nsamples, self.npars = np.shape(posterior)

        self.pnames = pnames
        self.bestp = bestp
        self.thinning = thinning
        self.ranges = ranges
        self.theme = theme
        if rect is None:
            rect = (0.1, 0.1, 0.98, 0.98)
        self.rect = rect
        if figsize is None:
            self.figsize = (8,8)
        self.plot_marginal = plot_marginal
        self.margin = margin
        self.ymargin = ymargin
        self.quantile = quantile
        self.bins = bins
        self.nlevels = nlevels
        #self.ms = ms
        self.fontsize = fontsize
        self.linewidth = linewidth
        self.orientation = 'vertical'


    def update(self):
        self.plot(
            self.plot_marginal, fignum=self.fignum,
        )

    def plot(
            self, plot_marginal=True, fignum=None,
        ):
        """Pairwise plus histogram plot."""
        npars = self.npars
        # Create new figure unless explicitly point to an existing one:
        if fignum is not None and plt.fignum_exists(fignum):
            self.fig = plt.figure(fignum)
        else:
            self.fig = plt.figure(fignum, self.figsize)
        self.fignum = self.fig.number

        # Define the axes:
        if self.pair_axes is None:
            self.pair_axes = np.tile(None, (npars-1, npars-1))
        nx = npars + int(self.plot_marginal) - 1

        for icol in range(npars-1):
            for irow in range(icol, npars-1):
                h = nx*irow + icol + 1 + npars*int(self.plot_marginal)
                self.pair_axes[irow,icol] = subplot(
                    self.rect, self.margin, h, nx, ymargin=self.ymargin)

        self.palette = copy.copy(self.theme['colormap'])
        self.palette.set_under(color='w')
        self.palette.set_bad(color='w')

        absolute_dens = False
        self.hist_xran, self.hist, self.lmax = hist_2D(
            self.posterior, self.ranges, self.bins, self.nlevels,
        )

        # The pair-wise data:
        _pairwise(
            self.hist, self.hist_xran,
            self.pair_axes, self.ranges, self.bestp,
            self.palette,
            self.nlevels,
            absolute_dens,
            self.lmax,
        )
        # The colorbar:
        dx = (self.rect[2]-self.rect[0])*0.03
        dy = (self.rect[3]-self.rect[1])*0.45
        self.colorbar = mpl.colorbar.Colorbar(
            plt.axes([self.rect[2]-dx, self.rect[3]-dy, dx, dy]),
        )

        # Marginal posterior:
        if self.hist_axes is None:
            self.hist_axes = np.tile(None, npars)
        for i in range(npars):
            h = (npars+1)*i + 1
            self.hist_axes[i] = subplot(
                self.rect, self.margin, h, npars, ymargin=self.ymargin)

        if plot_marginal:
            if self.quantile is None:
                pass
            elif self.source.pdf[0] is None:
                for i in range(npars):
                    pdf, xpdf, hpd_min = ms.cred_region(
                        self.posterior[:,i],
                        self.quantile, self.source.pdf[i], self.source.xpdf[i],
                    )
                    self.source.pdf[i] = pdf
                    self.source.xpdf[i] = xpdf

            yscale = False
            _histogram(
                self.posterior, self.bestp, self.ranges,
                self.hist_axes, self.bins,
                self.quantile, self.source.pdf, self.source.xpdf,
                self.linewidth, self.theme,
                yscale, self.orientation,
            )
        _plot_pairwise(self)


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
        print(f'Sharing updated value of {var_name} to {value}')
        setattr(obj, priv_name, value)
        #for fig in obj.figures:
        for i in reversed(range(len(obj.figures))):
            fig = obj.figures[i]
            if not is_open(fig.fig):
                obj.figures.pop(i)
                print(f'pop {i} {fig}')
            else:
                setattr(fig, var_name, value)


class Posterior(object):
    """Classification of posterior plotting tools.

    Examples
    --------
    >>> from importlib import reload
    >>> import mc3
    >>> import plots_concept as c
    >>> mcmc = np.load('MCMC_HD209458b_sing_0.29-2.0um_MM2017.npz')
    >>> posterior, zchain, zmask = mc3.utils.burn(mcmc)
    >>> idx = np.arange(7, 13)
    >>> post = posterior[:,idx]
    >>> pnames = mcmc['texnames'][idx]
    >>> bestp = mcmc['bestp'][idx]

    >>> reload(c)
    >>> p = c.Posterior(post, pnames, bestp)
    >>> f = p.plot()

    >>> new_pnames = [
           '$\\log_{10}(X_{\\rm Na})$',  '$\\log_{10}(X_{\\rm K})$',
           '$\\log_{10}(X_{\\rm H2O})$', '$\\log_{10}(X_{\\rm CH4})$',
           '$\\log_{10}(X_{\\rm NH3})$', '$\\log_{10}(X_{\\rm HCN})$']
    >>> p.pnames = new_pnames  # Auto-updates
    >>> p.pnames[0] = '$\\log_{10}(Q_{\\rm Na})$' # Does not auto-update
    """
    # Soft-update properties:
    pnames = ShareUpdate()
    bestp = ShareUpdate()
    ranges = ShareUpdate()
    thinning = ShareUpdate()
    theme = ShareUpdate()

    def __init__(
            self, posterior, pnames=None, bestp=None, ranges=None,
            thinning=1, quantile=0.683,
            bins=25, nlevels=20, fontsize=11, linewidth=1.5,
            pdf=None, xpdf=None,
            theme='default', orientation='vertical',
        ):
        self.figures = []
        # TBD: enforce posterior as 2D
        self.input_posterior = posterior
        self.thinning = thinning
        nsamples, self.npars = np.shape(posterior)

        # Defaults:
        if pnames is None:
            pnames = [f'p{i:02d}' for i in range(self.npars)]

        self.fig = None
        self.pnames = pnames
        self.bestp = bestp
        self.ranges = ranges
        self.quantile = quantile
        self.bins = bins
        self.nlevels = nlevels
        self.fontsize = fontsize
        self.linewidth = linewidth
        self.theme = theme
        self.orientation = orientation

        if pdf is None or xpdf is None:
            self.pdf = [None]*self.npars
            self.xpdf = [None]*self.npars
        else:
            self.pdf = pdf
            self.xpdf = xpdf

    def plot(self, plot_marginal=True, fignum=None):
        """
        Defaults to histogram plus pairwise
        """
        fig = Figure(
            self,
            self.input_posterior, self.pnames, self.bestp,
            self.ranges, self.theme,
            plot_marginal=plot_marginal,
        )
        self.figures.append(fig)
        fig.plot()
        return fig

    def plot_histogram(self, fignum=None, axes=None, quantile=None
        ):
        """Plot the marginal histograms of the posterior distribution
        >>> import mc3
        >>> import mc3.stats as ms
        >>> from mc3.plots import subplot
        >>> mcmc = np.load('MCMC_HD209458b_sing_0.29-2.0um_MM2017.npz')
        >>> posterior, zchain, zmask = mc3.utils.burn(mcmc)

        >>> idx = np.arange(0, 13)
        >>> idx = np.arange(7, 13)
        >>> #idx = np.arange(8, 13)
        >>> post = posterior[:,idx]
        >>> pnames = mcmc['texnames'][idx]
        >>> bestp = mcmc['bestp'][idx]
        >>> self = p = mc3.plots.Posterior(post, pnames, bestp)
        >>> p.plot_histogram()
        """
        # Create new figure unless explicitely point to an existing one:
        if fignum is not None and plt.fignum_exists(fignum):
            self.fig = plt.figure(fignum)
        else:
            self.fig = plt.figure(fignum, self.figsize)
        self.hist_fignum = self.fig.number

        if self.hist_axes is None:
            self.hist_axes = np.tile(None, self.npars)
        # Set number of rows:
        if self.npars < 6:  # Single row, N columns
            nx = self.npars
        elif self.npars < 13:  # Two rows, up to 6 columns
            nx = (self.npars+1) // 2
        elif self.npars < 25:  # Six columns, up to 4 rows
            nx = 6
        else:  # Stick with 4 rows,
            nx = 1 + (self.npars-1) // 4
        ny = 1 + (self.npars-1) // nx

        # Layout sizes:
        dx0 = 0.4
        size = dx0 + 1.45*nx, 2.0*ny
        self.hist_ymargin = 0.3 / ny
        self.fig.set_size_inches(*list(size))
        self.hist_axes = np.tile(None, self.npars)
        self.fig.clf()
        self.hist_rect[0] = dx0/size[0]
        self.hist_rect[1] = self.hist_ymargin

        for i in range(self.npars):
            ax = self.hist_axes[i] = subplot(
                self.hist_rect, self.margin, i+1, nx, ny, self.hist_ymargin)

        if axes is None:
            fig = plt.figure(self.hist_fignum, figsize=self.hist_figsize)
            fig.clf()
            for i in range(self.npars):
                ax = self.hist_axes[i] = subplot(
                    self.hist_rect, self.margin, i+1, nx, ny, self.hist_ymargin)
                if i%nx == 0 or self.orientation == 'horizontal':
                    ax.get_yaxis().set_label_text('Posterior', fontsize=self.fontsize)

        if self.quantile is None:
            pass
        #elif quantile != self.quantile or self.pdf[0] is None:
        elif self.pdf[0] is None:
            for i in range(self.npars):
                self.pdf[i], self.xpdf[i], hpd_min = ms.cred_region(
                    self.posterior[:,i], self.quantile,
                    self.pdf[i], self.xpdf[i])
        #self.quantile = quantile

        yscale = False
        _histogram(
            self.posterior, self.pnames, self.bestp, self.ranges,
            self.hist_axes,
            self.bins, self.quantile, self.pdf, self.xpdf,
            self.linewidth, self.fontsize, self.theme,
            yscale, self.orientation)

        if self.savefile is not None:
            fig.savefig(self.savefile, bbox_inches='tight')


    def add():
        """Add another posterior"""
        pass


    def update(self, **kwargs):
        for key, value in kwargs.items():
            # If key in valid keys:
            print(f'{key} = {value}')
            # Else: throw warning
        replot = False
        # if hard-replotting, update parameter in kwargs.keys():
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

    def plot_trace():
        pass


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Deprecated functions:

def trace(
        posterior, zchain=None, pnames=None, thinning=1,
        burnin=0, fignum=1000, savefile=None, fmt=".", ms=2.5, fs=11,
    ):
    # Deprecated function
    pass


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

