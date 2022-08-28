# Copyright (c) 2015-2022 Patricio Cubillos and contributors.
# mc3 is open-source software under the MIT license (see LICENSE).

__all__ = [
    'alphatize',
    'color_theme',
]

import numpy as np
from matplotlib.colors import is_color_like, to_rgb, ListedColormap


def alphatize(colors, alpha, background='w'):
    """
    Get RGB representation of a color as if it had the specified alpha.

    Parameters
    ----------
    colors: color or iterable of colors
        The color to alphatize.
    alpha: Float
        Alpha value to apply.
    background: color
        Background color.

    Returns
    -------
    rgb: RGB or list of RGB color arrays
        The RGB representation of the alphatized color (or list of colors).

    Examples
    --------
    >>> import mc3.plots as mp

    >>> # As string:
    >>> color = 'red'
    >>> alpha = 0.5
    >>> mp.alphatize(color, alpha)
    array([1. , 0.5, 0.5])

    >>> # As RGB tuple:
    >>> color = (1.0, 0.0, 0.0)
    >>> mp.alphatize(color, alpha)
    array([1. , 0.5, 0.5])

    >>> # Specify 'background':
    >>> color1 = 'red'
    >>> color2 = 'blue'
    >>> mp.alphatize(color1, alpha, color2)
    array([0.5, 0. , 0.5])

    >>> # Input a list of colors:
    >>> mp.alphatize(['r', 'b'], alpha=0.8)
    [array([1. , 0.2, 0.2]), array([0.2, 0.2, 1. ])]
    """
    flatten = False
    if is_color_like(colors):
        colors = [colors]
        flatten = True
    colors = [np.array(to_rgb(color)) for color in colors]
    background = np.array(to_rgb(background))

    # https://matplotlib.org/tutorials/colors/colors.html
    rgb = [(1.0-alpha) * background + alpha*c for c in colors]

    if flatten:
        return rgb[0]
    return rgb


def color_theme(color):
    """
    Generate a monochromatic color theme from given color.

    Parameters
    ----------
    color: color or iterable of colors
        The color to alphatize.

    Returns
    -------
    color_theme: Dict
        A dictionary containing sets of colors.

    Examples
    --------
    >>> import mc3.plots as mp
    >>> theme = mc3.plots.color_theme('xkcd:blue')
    """
    whites = [
        alphatize(color, alpha, 'white')
        for alpha in np.linspace(0.15, 1.0, 162)
    ]
    darks = [
        alphatize(color, alpha, 'black')
        for alpha in np.linspace(1.0, 0.50,  95)
    ]
    colormap = ListedColormap(whites + darks[1:])
    colormap.set_under(color='white')
    colormap.set_bad(color='white')

    color_theme = {
        'edgecolor': color,
        'facecolor': alphatize(color, 0.75, 'white'),
        'color': alphatize(color, 0.5, 'black'),
        'colormap': colormap,
    }
    return color_theme

