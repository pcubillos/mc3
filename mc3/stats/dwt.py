# Copyright (c) 2015-2026 Patricio Cubillos and contributors.
# mc3 is open-source software under the MIT license (see LICENSE).

__all__ = [
    'daubechies4',
    'dwt_daub4',
    'dwt_chisq',
]

import numpy as np


def daubechies4(array, n, inverse):
    """
    Applies the Daubechies-4 discrete wavelet transform (DWT)
    or its inverse (transpose filter) depending on isign.

    Parameters
    ----------
    a : np.ndarray
        Input signal (1D array of length n).
        Modified in place.
    n: int
        Hierarchy level of the transform.
    inverse: bool
        Calculate forward (False) or inverse (True) DWT
    """
    C0 =  0.48296291314453410
    C1 =  0.83651630373780772
    C2 =  0.22414386804201339
    C3 = -0.12940952255126034

    if len(array) < 4:
        return

    dwt = np.zeros(n)
    k = n >> 1  # n/2 using bit shift

    if inverse:
        dwt[0] = C2*array[k-1] + C1*array[n-1] + C0*array[0] + C3*array[k]
        dwt[1] = C3*array[k-1] - C0*array[n-1] + C1*array[0] - C2*array[k]
        j = 2
        for i in range(0, k-1):
            dwt[j]   = C2*array[i] + C1*array[i+k] + C0*array[i+1] + C3*array[i+k+1]
            dwt[j+1] = C3*array[i] - C0*array[i+k] + C1*array[i+1] - C2*array[i+k+1]
            j += 2
    else:  # Forward transform
        i = 0
        for j in range(0, n-3, 2):
            dwt[i]   = C0*array[j] + C1*array[j+1] + C2*array[j+2] + C3*array[j+3]
            dwt[i+k] = C3*array[j] - C2*array[j+1] + C1*array[j+2] - C0*array[j+3]
            i += 1
        # Wrap-around (circular convolution)
        dwt[i]   = C0*array[n-2] + C1*array[n-1] + C2*array[0] + C3*array[1]
        dwt[i+k] = C3*array[n-2] - C2*array[n-1] + C1*array[0] - C0*array[1]

    # Copy back into input array
    array[0:n] = dwt


def dwt_daub4(array, inverse=False):
    """
    1D discrete wavelet transform using the Daubechies 4-parameter wavelet
    Implements the pyramid algorithm replacing array by its DWT or inverse DWT

    Parameters
    ----------
    array: 1D float array
        Input data array to which to apply the DWT (modified in place)
    inverse: bool
        Calculate forward (False) or inverse (True) DWT

    Notes
    -----
    The input vector must have length 2**M with M an integer, otherwise
    the output will zero-padded to the next size of the form 2**M.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import mc3.stats as ms

    >>> # Calculate the inverse DWT for a unit vector:
    >>> nx = 1000
    >>> array = np.zeros(nx)
    >>> array[10:20] = 1.0
    >>> inverse = True
    >>> idwt = ms.dwt_daub4(array, inverse=inverse)
    >>> dwt_array = dawt(array, inverse=inverse)

    >>> # Plot the inverse DWT
    >>> fig = plt.figure(10)
    >>> fig.set_size_inches(7,4)
    >>> plt.clf()
    >>> plt.subplots_adjust(0.1, 0.1, 0.99, 0.96)
    >>> plt.plot(1-dwt_array/idwt, label='py DWT', color='k')
    >>> plt.plot(0.1*array, label='signal', color='tomato')
    >>> plt.plot(idwt, label='DWT', color='xkcd:blue')
    >>> plt.plot(dwt_array, label='py DWT', color='limegreen', dashes=(8,3))
    """
    size = len(array)
    M = np.ceil(np.log2(size))
    dwt_size = 2**int(M)

    dwt_array = np.zeros(dwt_size)
    dwt_array[0:size] = array

    if size < 4:
        return dwt_array

    if inverse:
        # Inverse transform: start from smallest hierarchy to largest
        n = 4
        while n <= dwt_size:
            daubechies4(dwt_array, n, inverse)
            n <<= 1
    else:
        # Forward transform: start from largest hierarchy to smallest
        n = dwt_size
        while n >= 4:
            daubechies4(dwt_array, n, inverse)
            n >>= 1

    return dwt_array


def dwt_chisq(model, data, params, priors=None, priorlow=None, priorup=None):
    """
    Calculate -2*ln(likelihood) in a wavelet-base (a pseudo chi-squared)
    based on Carter & Winn (2009), ApJ 704, 51.

    Parameters
    ----------
    model: 1D ndarray
        Model fit of data.
    data: 1D ndarray
        Data set array fitted by model.
    params: 1D float ndarray
        Model parameters (including the tree noise parameters: gamma,
        sigma_r, sigma_w; which must be the last three elements in params).
    priors: 1D ndarray
        Parameter prior values.
    priorlow: 1D ndarray
        Left-sided prior standard deviation (param < prior).
        A priorlow value of zero denotes a uniform prior.
    priorup: 1D ndarray
        Right-sided prior standard deviation (prior < param).
        A priorup value of zero denotes a uniform prior.

    Returns
    -------
    chisq: Float
        Wavelet-based (pseudo) chi-squared.

    Notes
    -----
    - If the residuals array size is not of the form 2**N, the routine
    zero-pads the array until this condition is satisfied.
    - The current code only supports gamma=1.

    Examples
    --------
    >>> import mc3.stats as ms
    >>> import numpy as np
    >>> # Compute chi-squared for a given model fitting a data set:
    >>> data = np.array([2.0, 0.0, 3.0, -2.0, -1.0, 2.0, 2.0, 0.0])
    >>> model = np.ones(8)
    >>> data = np.array([2.0, 0.0, 3.0, -2.0, -1.0, 2.0, 2.0])
    >>> model = np.ones(len(data))
    >>> params = np.array([1.0, 0.1, 0.1])
    >>> chisq1 = ms.old_dwt_chisq(model, data, params)
    >>> chisq2 = ms.dwt_chisq(model, data, params)
    >>> print(chisq1, chisq2)
    1693.22308882

    >>> # Now, say this is a three-parameter model, with a Gaussian prior
    >>> # on the last parameter:
    >>> priors = np.array([1.0, 0.2, 0.3])
    >>> plow   = np.array([0.0, 0.0, 0.1])
    >>> pup    = np.array([0.0, 0.0, 0.1])
    >>> chisq = ms.dwt_chisq(model, data, params, priors, plow, pup)
    >>> print(chisq)
    1697.2230888243134
    """
    if len(params) < 3:
        raise ValueError('Wavelet chisq should have at least three parameters')

    # g-factor of the covariance of the wavelet coefficients:
    # g(gamma=1) = 1.0 / (2*ln(2))
    g = 0.72134752

    size = len(data)
    M = int(np.ceil(np.log2(size)))

    # Expand (zero padding) to res to a size proportional to 2**M (zero padding)
    residuals = np.zeros(2**M)
    residuals[0:size] = data - model
    dwt_residuals = dwt_daub4(residuals, inverse=False)
    print("".join([f'{w:.8f}  ' for w in dwt_residuals]))

    gamma, sigma_r, sigma_w = params[-3:]
    # Equation (34) of CW2009, square of sigma_S
    sigma_S = g * sigma_r**2.0 * 2.0**-gamma + sigma_w**2.0
    # Second term in right-hand side of equation 32
    chisq = (
        (dwt_residuals[0]**2.0 + dwt_residuals[1]**2.0) / sigma_S +
        2.0*np.log(2*np.pi*sigma_S)
    )

    # Equation (33) of CW2009, sigma_W squared
    for m in range(1, M): # Number of scales
        sigma_W = sigma_r**2.0 * 2.0**(-gamma*m) + sigma_w**2.0
        n = 2**m
        res_sq = np.sum(dwt_residuals[n:2*n]**2.0)
        chisq += res_sq/sigma_W + n*np.log(2*np.pi*sigma_W)

    # Add priors contribution
    if priors is None or priorlow is None or priorup is None:
        return chisq

    delta_priors = params - priors
    for d_prior, e_low, e_up in zip(delta_priors, priorlow, priorup):
        # No prior
        if e_low<=0 or e_up<=0:
            continue
        # Gaussian prior
        if d_prior > 0:
            chisq += (d_prior/e_up)**2.0
        else:
            chisq += (d_prior/e_low)**2.0

    return chisq

