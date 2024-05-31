# Copyright (c) 2015-2023 Patricio Cubillos and contributors.
# mc3 is open-source software under the MIT license (see LICENSE).

__all__ = [
    'mcmc',
]

import platform
import time
import ctypes
import multiprocessing as mpr

import numpy as np

from . import chain as ch
from . import stats as ms


def mcmc(
        data, uncert, func, params, indparams, indparams_dict,
        pmin, pmax, pstep,
        prior, priorlow, priorup,
        nchains, ncpu, nsamples, sampler,
        wlike, fit_output, grtest, grbreak, grnmin, burnin, thinning,
        fgamma, fepsilon, hsize, kickoff, savefile, resume, log,
        pnames, texnames,
    ):
    """
    Mid-level routine called by mc3.sample() to execute Markov-chain Monte
    Carlo run.

    Parameters
    ----------
    data: 1D float ndarray
        Data to be fit by func.
    uncert: 1D float ndarray
        Uncertainties of data.
    func: Callable or string-iterable
        The callable function that models data as:
        model = func(params, *indparams, **indparams_dict)
    params: 1D float ndarray
        Set of initial fitting parameters for func.
    indparams: tuple
        Additional arguments required by func.
    indparams_dict: dict
        Additional keyword arguments required by func (if required).
    pmin: 1D ndarray
        Lower boundaries for the posterior exploration.
    pmax: 1D ndarray
        Upper boundaries for the posterior exploration.
    pstep: 1D ndarray
        Parameter stepping.
    prior: 1D ndarray
        Parameter prior distribution means.
    priorlow: 1D ndarray
        Lower prior uncertainty values.
    priorup: 1D ndarray
        Upper prior uncertainty values.
    nchains: Scalar
        Number of simultaneous chains to run.
    ncpu: Integer
        Number of processors for the MCMC chains.
    nsamples: Scalar
        Total number of samples.
    sampler: String
        MCMC sampling algorithm select from [mrw, demc, snooker]
    wlike: Bool
        If True, calculate the likelihood in a wavelet-base.
    grtest: Boolean
        Run Gelman & Rubin test.
    grbreak: Float
        Gelman-Rubin convergence threshold to stop the MCMC.
    grnmin: Float
        Minimum number of samples required for grbreak to stop the MCMC.
    burnin: Integer
        Number of burned-in (discarded) iterations.
    thinning: Integer
        Thinning factor of the chains (use every thinning-th iteration).
    fgamma: Float
        Proposals jump scale factor for DEMC's gamma.
    fepsilon: Float
        Jump scale factor for DEMC's support distribution.
    hsize: Integer
        Number of initial samples per chain.
    kickoff: String
        How to start the chains:
        'normal' for normal distribution around initial guess, or
        'uniform' for uniform distribution withing the given boundaries.
    savefile: String
        If not None, filename to store allparams and other MCMC results.
    resume: Boolean
        If True resume a previous run.
    log: mc3.utils.Log instance
        Logging object.

    Returns
    -------
    output: Dict
        A Dictionary containing the MCMC posterior distribution and related
        stats, including:
        - posterior: thinned posterior distribution of shape [nsamples, nfree].
        - zchain: chain indices for each sample in Z.
        - zmask: indices that turn Z into the desired posterior (remove burn-in)
        - chisq: chi^2 value for each sample in Z.
        - log_posterior: log(posterior) for the samples in Z.
        - burnin: number of burned-in samples per chain.
        - bestp: model parameters for the optimal log(posterior) sample.
        - best_model: model evaluated at bestp.
        - best_chisq: chi^2 for the optimal log(posterior) in the sample.
        - best_log_post: optimal log(posterior) in posterior.
        - acceptance_rate: sample's acceptance rate.

    Examples
    --------
    >>> # See https://mc3.readthedocs.io/en/latest/mcmc_tutorial.html
    """
    nfree  = int(np.sum(pstep > 0))
    ifree  = np.where(pstep > 0)[0]
    ishare = np.where(pstep < 0)[0]

    if resume:
        oldrun = np.load(savefile)
        zold = oldrun["posterior"]
        zchain_old = oldrun["zchain"]
        # Size of posterior (prior to this MCMC sample):
        pre_zsize = M0 = np.shape(zold)[0]
    else:
        pre_zsize = M0 = hsize*nchains

    # Number of Z samples per chain:
    nzchain = int(np.ceil(nsamples/nchains/thinning))
    # Number of iterations per chain:
    niter = nzchain * thinning
    # Total number of Z samples (initial + chains):
    zlen = pre_zsize + nzchain*nchains

    burnin = int(burnin)
    if not resume and niter < burnin:
        log.error(
            f"The number of burned-in samples ({burnin}) is greater than "
            f"the number of iterations per chain ({niter})"
        )

    # Initialize shared-memory variables:
    sm_freepars = mpr.Array(ctypes.c_double, nchains*nfree)
    freepars = np.ctypeslib.as_array(sm_freepars.get_obj())
    freepars = freepars.reshape((nchains, nfree))

    best_log_post = mpr.Value(ctypes.c_double, np.inf)
    sm_bestp = mpr.Array(ctypes.c_double, np.copy(params))
    bestp = np.ctypeslib.as_array(sm_bestp.get_obj())
    # There seems to be a strange behavior with np.ctypeslib.as_array()
    # when the argument is a single-element array. In this case, the
    # returned value is a two-dimensional array, instead of 1D. The
    # following line fixes(?) that behavior:
    if np.ndim(bestp) > 1:
        bestp = bestp.flatten()

    numaccept = mpr.Value(ctypes.c_int, 0)
    outbounds = mpr.Array(ctypes.c_int, nfree)

    # Z array with the chains history:
    sm_Z = mpr.Array(ctypes.c_double, zlen*nfree)
    Z = np.ctypeslib.as_array(sm_Z.get_obj())
    Z = Z.reshape((zlen, nfree))

    # Chi-square value of Z:
    sm_log_post = mpr.Array(ctypes.c_double, zlen)
    log_post = np.ctypeslib.as_array(sm_log_post.get_obj())
    # Chain index for given state in the Z array:
    sm_zchain = mpr.Array(ctypes.c_int, -np.ones(zlen, int))
    zchain = np.ctypeslib.as_array(sm_zchain.get_obj())
    # Current number of samples in the Z array:
    zsize = mpr.Value(ctypes.c_int, M0)
    # Burned samples in the Z array per chain:
    zburn = int(burnin/thinning)

    # Include values from previous run:
    if resume:
        Z[0:pre_zsize,:] = zold
        zchain[0:pre_zsize] = oldrun["zchain"]
        log_post[0:pre_zsize] = oldrun["log_post"]
        # Redefine zsize:
        zsize.value = pre_zsize
        numaccept.value = int(oldrun["acceptance_rate"] / 100. * pre_zsize)

    # Set GR N-min as number of thinned samples:
    if grnmin >= 1:
        grnmin = int(grnmin/thinning)
    elif grnmin > 0:
        grnmin = int(grnmin*nchains*(nzchain-zburn))
    elif grnmin < 0:
        log.error(
            "Invalid 'grnmin' argument (minimum number of samples to "
            "stop the MCMC under GR convergence), must either be grnmin > 1"
            "to set the minimum number of samples, or 0 < grnmin < 1"
            "to set the fraction of samples required to evaluate.")
    # Add these to compare grnmin to zsize (which also include them):
    grnmin += int(M0 + zburn*nchains)

    # Current length of each chain:
    sm_chainsize = mpr.Array(ctypes.c_int, np.tile(hsize, nchains))
    chainsize = np.ctypeslib.as_array(sm_chainsize.get_obj())

    # Number of chains per processor:
    ncpp = np.tile(int(nchains/ncpu), ncpu)
    ncpp[0:nchains % ncpu] += 1

    # Launch Chains:
    if platform.system() == 'Windows':
        multiprocess_context = mpr.get_context('spawn')
    else:
        multiprocess_context = mpr.get_context('fork')

    #mp_context = mpr.get_context('fork')
    pipes  = []
    chains = []
    for i in range(ncpu):
        p = multiprocess_context.Pipe()
        pipes.append(p[0])
        chains.append(
            ch.Chain(func, indparams, indparams_dict, p[1], data, uncert,
            params, freepars, pstep, pmin, pmax,
            sampler, wlike, prior, priorlow, priorup, thinning,
            fgamma, fepsilon, Z, zsize, log_post, zchain, M0,
            numaccept, outbounds, ncpp[i],
            chainsize, bestp, best_log_post, i, ncpu))

    if resume:
        bestp[:] = oldrun['bestp']
        best_log_post.value = oldrun['best_log_post']
        for c in range(nchains):
            chainsize[c] = np.sum(zchain_old==c)
    else:
        def random_pick(kickoff):
            x0 = np.copy(params[ifree])
            sigma = np.copy(pstep[ifree])
            x_min = np.copy(pmin[ifree])
            x_max = np.copy(pmax[ifree])
            while True:
                if kickoff == 'normal':
                    yield np.random.normal(x0, sigma)
                elif kickoff == 'uniform':
                    yield np.random.uniform(x_min, x_max)

        # Evaluate models for initial sample of Z:
        values = np.asarray(params)
        i = 0
        j = 0
        nmax_trials = 100 * M0
        for trial in random_pick(kickoff):
            if i == M0 or j == nmax_trials:
                break

            values[ifree] = trial
            if np.any(values > pmax) or np.any(values < pmin):
                j += 1
                continue
            # Update shared parameters:
            for s in ishare:
                values[s] = values[-int(pstep[s])-1]
            chi_square = -0.5*chains[0].eval_model(values, ret='chisq')
            if not np.isfinite(chi_square):
                j += 1
                continue
            Z[i] = values[ifree]
            log_post[i] = chi_square
            i += 1

        if i < M0-1:
            log.error(
                'Cannot populate an initial sample set of parameters, try '
                'updating the parameters initial guess to avoid sampling '
                'beyond the parameter boundaries or where the model returns '
                'non-finite values.'
            )

        # Best-fitting values (so far):
        izbest = np.argmax(log_post[0:M0])
        best_log_post.value = log_post[izbest]
        bestp[ifree] = np.copy(Z[izbest])
        if fit_output is not None:
            bestp[:] = np.copy(fit_output['bestp'])
            best_log_post.value = fit_output['best_log_post']

    # The output dict:
    output = {
        'pnames': pnames,
        'texnames': texnames,
        'pstep': pstep,
        'ifree': ifree,
        'burnin': zburn,
    }

    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Start loop:
    print("Yippee Ki Yay Monte Carlo!")
    log.msg(f"Start MCMC chains  ({time.ctime()})")
    for chain in chains:
        chain.start()
    bit = bool(1)  # Dummy variable to send through pipe for DEMC
    # Intermediate steps to run GR test and print progress report:
    intsteps = (nzchain*nchains) / 10
    report = intsteps

    while True:
        # Proposal jump:
        if sampler == "demc":
            # Send and receive bit for DEMC synchronization:
            for pipe in pipes:
                pipe.send(bit)
            for pipe in pipes:
                _ = pipe.recv()

        # Print intermediate info:
        if (zsize.value-pre_zsize >= report) or (zsize.value == zlen):
            report += intsteps
            log.progressbar((zsize.value+1.0-pre_zsize)/(nzchain*nchains))

            out_of_bounds = np.asarray(outbounds[:])
            chisq = -2*best_log_post.value
            log.msg(
                f"Out-of-bound Trials:\n{out_of_bounds}\n"
                f"Best Parameters: (chisq={chisq:.4f})\n{bestp[ifree]}",
                width=80)

            # Save intermediate state:
            if savefile is not None:
                ms.update_output(output, chains[0], hsize)
                np.savez(savefile, **output)

            # Gelman-Rubin statistics:
            if grtest and np.all(chainsize > (zburn+hsize)):
                psrf = ms.gelman_rubin(Z, zchain, zburn)
                log.msg(
                    f"Gelman-Rubin statistics for free parameters:\n{psrf}",
                    width=80,
                )
                if np.all(psrf < 1.01):
                    log.msg("All parameters converged to within 1% of unity.")
                converged = (
                    grbreak > 0.0 and
                    np.all(psrf < grbreak) and
                    zsize.value > grnmin
                )
                if converged:
                    with zsize.get_lock():
                        zsize.value = zlen
                    log.msg(
                        "\nAll parameters satisfy the GR convergence "
                        f"threshold of {grbreak:g}, stopping the MCMC.")
                    break
            if zsize.value == zlen:
                break

    # Make sure chains finish and release all locks
    for chain in chains:
        chain.join()
        
    for chain in chains:  # Make sure to terminate the subprocesses
        chain.terminate()

    # Evaluate model for best fitting parameters:
    posterior = ms.update_output(output, chains[0], hsize)
    # DEBUG
    if posterior is None:
        posterior = np.zeros(1)

    # Print out Summary:
    Z = output['posterior']
    nsample = len(Z)*thinning
    nzsample = len(posterior)
    fmt = len(str(nsample))
    chain_iter = nsample // nchains
    accept_rate = output['acceptance_rate']

    log.msg('\nMCMC Summary:\n-------------')
    log.msg(
        f"Number of evaluated samples:        {nsample:{fmt}d}\n"
        f"Number of parallel chains:          {nchains:{fmt}d}\n"
        f"Average iterations per chain:       {chain_iter:{fmt}d}\n"
        f"Burned-in iterations per chain:     {burnin:{fmt}d}\n"
        f"Thinning factor:                    {thinning:{fmt}d}\n"
        f"MCMC sample size (thinned, burned): {nzsample:{fmt}d}\n"
        f"Acceptance rate:   {accept_rate:.2f}%\n", indent=2)

    return output
