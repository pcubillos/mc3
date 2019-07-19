# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

import os
import sys
import subprocess
import pytest

import numpy as np

import MCcubed as mc3


def quad(p, x):
    """
    Quadratic polynomial function.

    Parameters
        p: Polynomial constant, linear, and quadratic coefficients.
        x: Array of dependent variables where to evaluate the polynomial.
    Returns
        y: Polinomial evaluated at x:  y(x) = p0 + p1*x + p2*x^2
    """
    y = p[0] + p[1]*x + p[2]*x**2.0
    return y


np.random.seed(12)
# Create a synthetic dataset:
x = np.linspace(0, 10, 100)
p0 = [4.5, -2.4, 0.5]
y = quad(p0, x)
uncert = np.sqrt(np.abs(y))
error = np.random.normal(0, uncert)
data = y + error

p1 = [4.5, 4.5, 0.5]
y1 = quad(p1, x)
uncert1 = np.sqrt(np.abs(y1))
data1 = y1 + np.random.normal(0, uncert1)

# Fit the quad polynomial coefficients:
params   = np.array([10.0, -2.0, 0.1])  # Initial guess of fitting params.
pstep    = np.array([0.03, 0.03, 0.05])
pnames   = ["constant", "linear", "quadratic"]
texnames = ["$\\alpha$", "$\\log(\\beta)$", "quadratic"]
walk = 'snooker'

def test_minimal():
    output = mc3.mcmc(data, uncert, func=quad, walk=walk, indparams=[x],
        params=np.copy(params), pstep=pstep, nsamples=1e4, burnin=100)
    # No error? that's a pass.
    assert output is not None


def test_demc():
    output = mc3.mcmc(data, uncert, func=quad, indparams=[x],
        params=np.copy(params), pstep=pstep, nsamples=1e4, burnin=100,
        walk='demc')
    assert output is not None


def test_func_as_strings(tmp_path):
    p = tmp_path / "quadratic.py"
    CONTENT = u'def quad(p, x):\n  y = p[0] + p[1]*x + p[2]*x**2.0\n  return y'
    p.write_text(CONTENT)
    output = mc3.mcmc(data, uncert,
        func=('quad', 'quadratic', str(tmp_path)), walk=walk,
        indparams=[x], params=np.copy(params), pstep=pstep,
        nsamples=1e4, burnin=100)
    assert output is not None


def test_shared():
    output = mc3.mcmc(data1, uncert1, func=quad, walk=walk, indparams=[x],
        params=np.copy(params), pstep=[0.03, -1, 0.05],
        nsamples=1e4, burnin=100)
    assert output is not None
    assert output['bestp'][1] == output['bestp'][0]


def test_fixed():
    pars = np.copy(params)
    pars[0] = p0[0]
    output = mc3.mcmc(data, uncert, func=quad, walk=walk, indparams=[x],
        params=np.copy(params), pstep=[0, 0.03, 0.05], nsamples=1e4, burnin=100)
    assert output is not None
    assert len(output['bestp']) == len(params)
    assert output['bestp'][0] == params[0]
    assert output['CRlo'][0] == 0
    assert output['CRhi'][0] == 0
    assert output['stdp'][0] == 0


def test_bounds():
    output = mc3.mcmc(data, uncert, func=quad, walk=walk, indparams=[x],
        params=[4.5, -2.5, 0.5], pstep=pstep,
        pmin=[4.0, -3.0, 0.4], pmax=[5.0, -2.0, 0.6],
        nsamples=1e4, burnin=100)


def test_pnames(capsys):
    output = mc3.mcmc(data, uncert, func=quad, walk=walk, indparams=[x],
        params=np.copy(params), pstep=pstep, nsamples=1e4, burnin=100,
        pnames=pnames)
    captured = capsys.readouterr()
    assert output is not None
    assert "constant"  in captured.out
    assert "linear"    in captured.out
    assert "quadratic" in captured.out


def test_texnames(capsys):
    output = mc3.mcmc(data, uncert, func=quad, walk=walk, indparams=[x],
        params=np.copy(params), pstep=pstep, nsamples=1e4, burnin=100,
        texnames=texnames)
    captured = capsys.readouterr()
    assert output is not None
    assert "$\\alpha$"     in captured.out
    assert "$\\log(\\beta" in captured.out
    assert "quadratic"     in captured.out


def test_pnames_texnames(capsys):
    output = mc3.mcmc(data, uncert, func=quad, walk=walk, indparams=[x],
        params=np.copy(params), pstep=pstep, nsamples=1e4, burnin=100,
        pnames=pnames, texnames=texnames)
    captured = capsys.readouterr()
    assert output is not None
    assert "constant"  in captured.out
    assert "linear"    in captured.out
    assert "quadratic" in captured.out


@pytest.mark.parametrize('leastsq', ['lm', 'trf'])
def test_optimize(capsys, leastsq):
    output = mc3.mcmc(data, uncert, func=quad, walk=walk, indparams=[x],
        params=np.copy(params), pstep=pstep, nsamples=1e4, burnin=100,
        leastsq=leastsq)
    captured = capsys.readouterr()
    assert output is not None
    assert "Least-squares best-fitting parameters:" in captured.out
    np.testing.assert_allclose(output['bestp'],
        np.array([4.28263253, -2.40781859, 0.49534411]), rtol=1e-7)


def test_optimize_chisqscale(capsys):
    unc = np.copy(uncert)
    output = mc3.mcmc(data, uncert, func=quad, walk=walk, indparams=[x],
        params=np.copy(params), pstep=pstep, nsamples=1e4, burnin=100,
        leastsq='lm', chisqscale=True)
    captured = capsys.readouterr()
    assert output is not None
    assert "Least-squares best-fitting parameters (rescaled chisq):" \
        in captured.out
    assert "Reduced chi-squared:                1.0000" in captured.out
    # Assert that uncert has not mutated:
    np.testing.assert_equal(uncert, unc)


def test_gr(capsys):
    output = mc3.mcmc(data, uncert, func=quad, walk=walk, indparams=[x],
        params=np.copy(params), pstep=pstep, nsamples=1e4, burnin=100,
        grtest=True)
    captured = capsys.readouterr()
    assert output is not None
    assert "Gelman-Rubin statistics for free parameters:" in captured.out


def test_gr_break_frac(capsys):
    output = mc3.mcmc(data, uncert, func=quad, walk=walk, indparams=[x],
        params=np.copy(params), pstep=pstep, nsamples=1e4, burnin=100,
        grtest=True, grbreak=1.1, grnmin=0.51)
    captured = capsys.readouterr()
    assert output is not None
    assert "All parameters satisfy the GR convergence threshold of 1.1" \
           in captured.out


def test_gr_break_iterations(capsys):
    output = mc3.mcmc(data, uncert, func=quad, walk=walk, indparams=[x],
        params=np.copy(params), pstep=pstep, nsamples=1e4, burnin=100,
        grtest=True, grbreak=1.1, grnmin=5000.0)
    captured = capsys.readouterr()
    assert output is not None
    assert "All parameters satisfy the GR convergence threshold of 1.1" \
           in captured.out


def test_priors_gauss():
    prior    = np.array([ 4.5,  0.0,   0.0])
    priorlow = np.array([ 0.1,  0.0,   0.0])
    priorup  = np.array([ 0.1,  0.0,   0.0])
    output = mc3.mcmc(data, uncert, func=quad, walk=walk, indparams=[x],
        params=np.copy(params), pstep=pstep, nsamples=1e4, burnin=100,
        prior=prior, priorlow=priorlow, priorup=priorup)
    assert output is not None


def test_log(capsys, tmp_path):
    os.chdir(str(tmp_path))
    output = mc3.mcmc(data, uncert, func=quad, walk=walk, indparams=[x],
        params=np.copy(params), pstep=pstep, nsamples=1e4, burnin=100,
        log='MCMC.log')
    captured = capsys.readouterr()
    assert output is not None
    assert "'MCMC.log'" in captured.out
    assert "MCMC.log" in os.listdir(".")


def test_savefile(capsys, tmp_path):
    os.chdir(str(tmp_path))
    output = mc3.mcmc(data, uncert, func=quad, walk=walk, indparams=[x],
        params=np.copy(params), pstep=pstep, nsamples=1e4, burnin=100,
        savefile='MCMC.npz')
    captured = capsys.readouterr()
    assert output is not None
    assert "'MCMC.npz'" in captured.out
    assert "MCMC.npz" in os.listdir(".")


def test_plots(capsys, tmp_path):
    os.chdir(str(tmp_path))
    output = mc3.mcmc(data, uncert, func=quad, walk=walk, indparams=[x],
        params=np.copy(params), pstep=pstep, nsamples=1e4, burnin=100,
        plots=True)
    captured = capsys.readouterr()
    assert output is not None
    assert "'MCMC_trace.png'"     in captured.out
    assert "'MCMC_pairwise.png'"  in captured.out
    assert "'MCMC_posterior.png'" in captured.out
    assert "'MCMC_model.png'"     in captured.out
    assert "MCMC_trace.png"     in os.listdir(".")
    assert "MCMC_pairwise.png"  in os.listdir(".")
    assert "MCMC_posterior.png" in os.listdir(".")
    assert "MCMC_model.png"     in os.listdir(".")


# Now, trigger the errors:
def test_data_error(capsys):
    output = mc3.mcmc(uncert=uncert, func=quad, walk=walk, indparams=[x],
        params=np.copy(params), pstep=pstep, nsamples=1e4, burnin=100)
    captured = capsys.readouterr()
    assert output is None
    assert "'data' is a required argument." in captured.out


def test_walk_error(capsys):
    output = mc3.mcmc(data=data, uncert=uncert, func=quad,
        indparams=[x], params=np.copy(params), pstep=pstep,
        nsamples=1e4, burnin=100)
    captured = capsys.readouterr()
    assert output is None
    assert "'walk' is a required argument." in captured.out


def test_nsamples_error(capsys):
    output = mc3.mcmc(uncert=uncert, func=quad, walk=walk, indparams=[x],
        params=np.copy(params), pstep=pstep, burnin=100)
    captured = capsys.readouterr()
    assert output is None
    assert "'nsamples' is a required argument for MCMC runs." in captured.out


def test_uncert_error(capsys):
    output = mc3.mcmc(data=data, func=quad, walk=walk, indparams=[x],
        params=np.copy(params), pstep=pstep, nsamples=1e4, burnin=100)
    captured = capsys.readouterr()
    assert output is None
    assert "'uncert' is a required argument." in captured.out


def test_func_error(capsys):
    output = mc3.mcmc(data=data, uncert=uncert, walk=walk, indparams=[x],
        params=np.copy(params), pstep=pstep, nsamples=1e4, burnin=100)
    captured = capsys.readouterr()
    assert output is None
    assert "'func' must be either a callable or an iterable" in captured.out


def test_params_error(capsys):
    output = mc3.mcmc(data=data, uncert=uncert, func=quad, walk=walk,
        indparams=[x], pstep=pstep, nsamples=1e4, burnin=100)
    captured = capsys.readouterr()
    assert output is None
    assert "'params' is a required argument" in captured.out


def test_samples_error(capsys):
    output = mc3.mcmc(data=data, uncert=uncert, func=quad, walk=walk,
        indparams=[x], params=np.copy(params), pstep=pstep,
        nsamples=1e4, burnin=2000)
    captured = capsys.readouterr()
    assert output is None
    assert "The number of burned-in samples (2000) is greater" in captured.out


def test_leastsq_error(capsys):
    output = mc3.mcmc(data=data, uncert=uncert, func=quad, walk=walk,
        indparams=[x], params=np.copy(params), pstep=pstep,
        leastsq='invalid', nsamples=1e4, burnin=100)
    captured = capsys.readouterr()
    assert output is None
    assert "Invalid 'leastsq' input (invalid). Must select from " \
           "['lm', 'trf']." in captured.out


def test_entry_point_version(capfd):
    subprocess.call('mc3 -v'.split())
    if sys.version_info.major == 3:
        captured = capfd.readouterr().out
    else:
        captured = capfd.readouterr().err
    assert captured == 'MC3 version {:s}.\n'.format(mc3.__version__)


def test_entry_point(tmp_path):
    os.chdir(str(tmp_path))
    p = tmp_path / 'MCMC.cfg'
    p.write_text(u'''[MCMC]
data      = data.npz
indparams = indp.npz

func     = quad quadratic
params   =  10.0  -2.0   0.1
pmin     = -25.0 -10.0 -10.0
pmax     =  30.0  10.0  10.0
pstep    =   1.0   0.5   0.1

nsamples = 1e4
nchains  = 7
walk     = snooker
grtest   = True
burnin   = 100
plots    = True

savefile = MCMC_test.npz''')
    p = tmp_path / 'quadratic.py'
    p.write_text(u'''
def quad(p, x):
    y = p[0] + p[1]*x + p[2]*x**2.0
    return y''')
    print('\n' + str(os.listdir('.')))
    # Create synthetic dataset:
    from quadratic import quad
    x  = np.linspace(0, 10, 1000)         # Independent model variable
    p0 = [3, -2.4, 0.5]                   # True-underlying model parameters
    y  = quad(p0, x)                      # Noiseless model
    uncert = np.sqrt(np.abs(y))           # Data points uncertainty
    error  = np.random.normal(0, uncert)  # Noise for the data
    data   = y + error                    # Noisy data set
    # Store data set and other inputs:
    mc3.utils.savebin([data, uncert], 'data.npz')
    mc3.utils.savebin([x],            'indp.npz')
    subprocess.call('mc3 -c MCMC.cfg'.split())
    assert "MCMC_test.npz"           in os.listdir(".")
    assert "MCMC_test_trace.png"     in os.listdir(".")
    assert "MCMC_test_pairwise.png"  in os.listdir(".")
    assert "MCMC_test_posterior.png" in os.listdir(".")
    assert "MCMC_test_model.png"     in os.listdir(".")


def test_deprecation_nproc(capsys):
    output = mc3.mcmc(data=data, uncert=uncert, func=quad, walk=walk,
        indparams=[x], params=np.copy(params), pstep=pstep,
        nsamples=1e3, burnin=2, nproc=7)
    captured = capsys.readouterr()
    assert output is not None
    assert "nproc argument is deprecated. Use ncpu instead." \
        in captured.out


def test_deprecation_parname(capsys):
    output = mc3.mcmc(data=data, uncert=uncert, func=quad, walk=walk,
        indparams=[x], params=np.copy(params), pstep=pstep,
        nsamples=1e3, burnin=2, parname=pnames)
    captured = capsys.readouterr()
    assert output is not None
    assert "parname argument is deprecated. Use pnames instead." \
        in captured.out


def test_deprecation_stepsize(capsys):
    output = mc3.mcmc(data=data, uncert=uncert, func=quad, walk=walk,
        indparams=[x], params=np.copy(params), stepsize=pstep,
        nsamples=1e3, burnin=2, ncpu=7)
    captured = capsys.readouterr()
    assert output is not None
    assert "stepsize argument is deprecated. Use pstep instead." \
        in captured.out


def test_deprecation_chireturn(capsys):
    output = mc3.mcmc(data=data, uncert=uncert, func=quad, walk=walk,
        indparams=[x], params=np.copy(params), stepsize=pstep,
        nsamples=1e3, burnin=2, ncpu=7, chireturn=True)
    captured = capsys.readouterr()
    assert output is not None
    assert "chireturn argument is deprecated." in captured.out


def test_deprecation_full_output(capsys):
    output = mc3.mcmc(data=data, uncert=uncert, func=quad, walk=walk,
        indparams=[x], params=np.copy(params), stepsize=pstep,
        nsamples=1e3, burnin=2, ncpu=7, full_output=True)
    captured = capsys.readouterr()
    assert output is not None
    assert "full_output argument is deprecated." in captured.out


@pytest.mark.parametrize('lm', [None, True, False])
@pytest.mark.parametrize('leastsq', [None, True, False, 'lm'])
def test_deprecation_leastsq(capsys, lm, leastsq):
    if leastsq is True and lm is False:
        ls = 'trf'
    elif leastsq in [True, 'lm']:
        ls = 'lm'
    else:
        ls = None
    output = mc3.mcmc(data=data, uncert=uncert, func=quad, walk=walk,
        indparams=[x], params=np.copy(params), stepsize=pstep,
        nsamples=1e3, burnin=2, ncpu=7, leastsq=leastsq, lm=lm)
    captured = capsys.readouterr()
    assert output is not None
    if isinstance(leastsq, bool):
        assert "leastsq as boolean is deprecated.  See docs for new usage.  " \
           "Set\nleastsq={}".format(repr(ls)) in captured.out
    if isinstance(lm, bool):
        assert "lm argument is deprecated.  See new usage of leastsq.  " \
               "Set\nleastsq={}".format(repr(ls)) in captured.out

