# Copyright (c) 2015-2022 Patricio Cubillos and contributors.
# mc3 is open-source software under the MIT license (see LICENSE).

__all__ = [
    'alphatize',
    'Theme',
    'THEMES',
]

import numpy as np
from matplotlib.colors import (
    is_color_like,
    same_color,
    to_rgb,
    ListedColormap,
)


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


class Theme():
    """A monochromatic color theme from given color"""
    def __init__(self, color, alpha_light=0.15, alpha_dark=0.5):
        """
        Parameters
        ----------
        color: color or iterable of colors
            The color to alphatize.
        alpha_light: Float
            Alpha color value to merge with white to make self.light_color.
        alpha_dark: Float
            Alpha color value to merge with black.

        Examples
        --------
        >>> import mc3.plots.colors as colors
        >>> theme = colors.Theme('xkcd:blue')
        >>> theme = colors.Theme([0.0, 0.2, 0.8])
        """
        whites = [
            alphatize(color, alpha, 'white')
            for alpha in np.linspace(alpha_light, 1.0, 162)
        ]
        darks = [
            alphatize(color, alpha, 'black')
            for alpha in np.linspace(1.0, alpha_dark, 95)
        ]
        colormap = ListedColormap(whites + darks[1:])
        colormap.set_under(color='white')
        colormap.set_bad(color='white')

        self.light_color = alphatize(color, 0.75, 'white')
        self.color = color
        self.dark_color = alphatize(color, alpha_dark, 'black')
        self.colormap = colormap

    def __repr__(self):
        return f"Theme({repr(self.color)})"

    def __eq__(self, other):
        return (
            same_color(self.color, other.color) and
            same_color(self.light_color, other.light_color) and
            same_color(self.dark_color, other.dark_color) and
            self.colormap == other.colormap
        )


# Setup for THEMES:
yellow = alphatize('gold', 0.7, 'orange')
yellow_theme = Theme(yellow, alpha_light=0.2, alpha_dark=0.6)
yellow_theme.color = 'orange'
yellow_theme.light_color = 'gold'
yellow_theme.dark_color = 'darkgoldenrod'


THEMES = {
    'red': Theme('xkcd:tomato'),
    'orange': Theme('darkorange'),
    'yellow': yellow_theme,
    'green': Theme('xkcd:green'),
    'lightblue': Theme('dodgerblue'),
    'blue': Theme('xkcd:blue'),
    'purple': Theme('xkcd:violet'),
    'indigo': Theme('xkcd:indigo'),
    'black': Theme('0.3'),
}

