# Copyright (c) 2015-2021 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

__all__ = [
    "prayer_beads",
    ]

def prayer_beads(data=None, nprays=0):
    """
    Implement a prayer-bead method to estimate parameter uncertainties.

    Parameters
    ----------
    data: 1D float ndarray
        A time-series dataset.
    nprays: Integer
        Number of prayer-bead shifts.  If nprays=0, set to the number
        of data points.

    Notes
    -----
    Believing in a prayer bead is a mere act of faith, please don't
    do that, we are scientists for god's sake!
    """
    print(
        "Believing in prayer beads is a mere act of faith, please don't use it"
        "\nfor published articles (see Cubillos et al. 2017, AJ, 153).")
    return None
