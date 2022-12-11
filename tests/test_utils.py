# Copyright (c) 2015-2022 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

import pytest
import numpy as np
import mc3.utils as mu


# A Mock posterior:
Z0 = np.array([0, 1, 10, 20, 30, 11, 31, 21, 12, 22, 32], dtype=np.double)
zchain = np.array([-1, -1, 0, 1, 2, 0, 2, 1, 0, 1, 2])
npars = 1
Z = np.zeros((len(Z0), npars))
for i in range(npars):
    Z[:,i] = Z0 + 100*i


def test_parray_none():
    assert mu.parray('None') is None


def test_parray_empty():
    np.testing.assert_equal(mu.parray(''), np.array([]))


def test_parray_numbers():
    np.testing.assert_equal(mu.parray('1 2 3'),
        np.array([1.0, 2.0, 3.0], np.double))


def test_parray_strings():
    assert mu.parray("a b\nc") == ["a", "b", "c"]


def test_saveascii(tmp_path):
    asciifile = str(tmp_path / "saved_ascii.txt")
    a = np.arange(4) * np.pi
    b = np.arange(4)
    c = np.logspace(0, 12, 4)
    data = [a,b,c]
    mu.saveascii(data, asciifile)

    with open(asciifile, "r") as f:
        output = f.read()
    assert output == ('        0         0         1\n'
                      '3.1415927         1     10000\n'
                      '6.2831853         2     1e+08\n'
                      ' 9.424778         3     1e+12\n')

def test_loadascii(tmp_path):
    # loadascii() is equivalent to np.loadtxt(file, unpack=True)
    asciifile = str(tmp_path / "saved_ascii.txt")
    with open(asciifile, "w") as f:
        f.write("# comment\n"
                "        0         0         1\n"
                "3.1415927         1     10000\n"
                "\n"
                "6.2831853         2     1e+08\n"
                " 9.424778         3     1e+12\n")
    data=np.array([
       [0.0000000e+00, 3.1415927e+00, 6.2831853e+00, 9.4247780e+00],
       [0.0000000e+00, 1.0000000e+00, 2.0000000e+00, 3.0000000e+00],
       [1.0000000e+00, 1.0000000e+04, 1.0000000e+08, 1.0000000e+12]])
    np.testing.assert_equal(data, mu.loadascii(asciifile))


def test_load_savebin_array(tmp_path):
    binfile = str(tmp_path / 'saved_bin.npz')
    data = np.arange(4)
    indata = [data]
    mu.savebin(indata, binfile)
    outdata = mu.loadbin(binfile)
    assert type(outdata[0]) == np.ndarray
    np.testing.assert_equal(outdata[0], data)


@pytest.mark.parametrize('data', ['one', True, [42], (42,42)])
def test_load_savebin(tmp_path, data):
    binfile = str(tmp_path / 'saved_bin.npz')
    dtype = type(data)
    indata = [data]
    mu.savebin(indata, binfile)
    outdata = mu.loadbin(binfile)
    assert type(outdata[0]) == dtype
    np.testing.assert_equal(outdata[0], data)


def test_loadsavebin_all(tmp_path):
    # This could be replaced with pickle files
    binfile = str(tmp_path / "saved_bin.npz")
    indata = [np.arange(4), "one", np.ones((2,2)), True, [42], (42, 42)]
    mu.savebin(indata, binfile)
    outdata = mu.loadbin(binfile)
    # Check types:
    assert type(outdata[0]) == np.ndarray
    assert type(outdata[1]) == str
    assert type(outdata[2]) == np.ndarray
    assert type(outdata[3]) == bool
    assert type(outdata[4]) == list
    assert type(outdata[5]) == tuple
    # Check values:
    np.testing.assert_equal(outdata[0], np.arange(4))
    assert outdata[1] == 'one'
    np.testing.assert_equal(outdata[2], np.ones((2,2)))
    assert outdata[3] == True
    assert outdata[4] == [42]
    assert outdata[5] == (42,42)


@pytest.mark.skip()
def test_isfile():
    pass


def test_burn_Z_unburn():
    # Only remove pre-MCMC samples (zchain==-1):
    burnin = 0
    posterior, chain, mask = mu.burn(Z=Z, zchain=zchain, burnin=burnin)
    np.testing.assert_equal(posterior,
        np.array([[10., 11., 12., 20., 21., 22., 30., 31., 32.]]).T)
    np.testing.assert_equal(chain, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]))
    np.testing.assert_equal(mask,  np.array([2, 5, 8, 3, 7, 9, 4, 6, 10]))


def test_burn_Z():
    burnin = 1
    posterior, chain, mask = mu.burn(Z=Z, zchain=zchain, burnin=burnin)
    np.testing.assert_equal(posterior, np.array([[11.,12.,21.,22.,31.,32.]]).T)
    np.testing.assert_equal(chain, np.array([0, 0, 1, 1, 2, 2]))
    np.testing.assert_equal(mask,  np.array([5, 8, 7, 9, 6, 10]))


def test_burn_dict():
    Zdict = {'posterior':Z, 'zchain':zchain, 'burnin':1}
    posterior, chain, mask = mu.burn(Zdict)
    np.testing.assert_equal(posterior, np.array([[11.,12.,21.,22.,31.,32.]]).T)
    np.testing.assert_equal(chain, np.array([0, 0, 1, 1, 2, 2]))
    np.testing.assert_equal(mask,  np.array([5, 8, 7, 9, 6, 10]))


def test_burn_unsort():
    Zdict = {'posterior':Z, 'zchain':zchain, 'burnin':1}
    posterior, chain, mask = mu.burn(Zdict, sort=False)
    np.testing.assert_equal(posterior, np.array([[11.,31.,21.,12.,22.,32.]]).T)
    np.testing.assert_equal(chain, np.array([0, 2, 1, 0, 1, 2]))
    np.testing.assert_equal(mask,  np.array([5, 6, 7, 8, 9, 10]))


def test_burn_override_burnin():
    Zdict = {'posterior':Z, 'zchain':zchain, 'burnin':1}
    posterior, chain, mask = mu.burn(Zdict, burnin=0)
    expected_post = np.array([[10., 11., 12., 20., 21., 22., 30., 31., 32.]]).T

    np.testing.assert_equal(posterior, expected_post)
    np.testing.assert_equal(chain, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]))
    np.testing.assert_equal(mask,  np.array([2, 5, 8, 3, 7, 9, 4, 6, 10]))


def test_parnames():
    expected_pnames = np.array(['Param 1', 'Param 2', 'Param 3'])
    np.testing.assert_equal(mu.default_parnames(3), expected_pnames)


def test_tex_parameters_default():
    values    = [9.29185155e+02, -3.25725507e+00, 8.80628658e-01]
    lo_bounds = [5.29185155e+02, -4.02435791e+00, 6.43578351e-01]
    hi_bounds = [1.43406714e+03, -2.76718364e+00, 9.87000918e-01]

    expected_texs = [
        '$929.2^{+504.9}_{-400.0}$',
        '$-3.26^{+0.49}_{-0.77}$',
        '$0.88^{+0.11}_{-0.24}$',
    ]
    tex_vals = mu.tex_parameters(values, lo_bounds, hi_bounds)
    for tex, expected in zip(tex_vals, expected_texs):
        assert tex == expected


def test_tex_parameters_digits1():
    values    = [9.29185155e+02, -3.25725507e+00, 8.80628658e-01]
    lo_bounds = [5.29185155e+02, -4.02435791e+00, 6.43578351e-01]
    hi_bounds = [1.43406714e+03, -2.76718364e+00, 9.87000918e-01]

    expected_texs = [
        '$929.2^{+504.9}_{-400.0}$',
        '$-3.3^{+0.5}_{-0.8}$',
        '$0.9^{+0.1}_{-0.2}$',
    ]
    tex_vals = mu.tex_parameters(
        values, lo_bounds, hi_bounds, significant_digits=1,
    )
    for tex, expected in zip(tex_vals, expected_texs):
        assert tex == expected


def test_tex_parameters_digits3():
    values    = [9.29185155e+02, -3.25725507e+00, 8.80628658e-01]
    lo_bounds = [5.29185155e+02, -4.02435791e+00, 6.43578351e-01]
    hi_bounds = [1.43406714e+03, -2.76718364e+00, 9.87000918e-01]

    expected_texs = [
        '$929.2^{+504.9}_{-400.0}$',
        '$-3.257^{+0.490}_{-0.767}$',
        '$0.881^{+0.106}_{-0.237}$',
    ]
    tex_vals = mu.tex_parameters(
        values, lo_bounds, hi_bounds, significant_digits=3,
    )
    for tex, expected in zip(tex_vals, expected_texs):
        assert tex == expected


def test_tex_parameters_math_names():
    values    = [9.29185155e+02, -3.25725507e+00, 8.80628658e-01]
    lo_bounds = [5.29185155e+02, -4.02435791e+00, 6.43578351e-01]
    hi_bounds = [1.43406714e+03, -2.76718364e+00, 9.87000918e-01]
    names = [
        r'$T_{\rm iso}$', r'$\log\,X_{\rm H2O}$', r'$\phi_{\rm patchy}$',
    ]

    expected_texs = [
        r'$T_{\rm iso} = 929.2^{+504.9}_{-400.0}$',
        r'$\log\,X_{\rm H2O} = -3.26^{+0.49}_{-0.77}$',
        r'$\phi_{\rm patchy} = 0.88^{+0.11}_{-0.24}$',
    ]
    tex_vals = mu.tex_parameters(values, lo_bounds, hi_bounds, names)
    for tex, expected in zip(tex_vals, expected_texs):
        assert tex == expected


def test_tex_parameters_plain_names():
    values    = [9.29185155e+02, -3.25725507e+00, 8.80628658e-01]
    lo_bounds = [5.29185155e+02, -4.02435791e+00, 6.43578351e-01]
    hi_bounds = [1.43406714e+03, -2.76718364e+00, 9.87000918e-01]
    names = ['Tiso', 'log H2O', 'phi_patchy']

    expected_texs = [
        r'Tiso$ = 929.2^{+504.9}_{-400.0}$',
        r'log H2O$ = -3.26^{+0.49}_{-0.77}$',
        r'phi_patchy$ = 0.88^{+0.11}_{-0.24}$',
    ]
    tex_vals = mu.tex_parameters(values, lo_bounds, hi_bounds, names)
    for tex, expected in zip(tex_vals, expected_texs):
        assert tex == expected


@pytest.mark.parametrize('value', [None, np.nan])
def test_tex_parameters_none_value(value):
    values    = [value,          -3.25725507e+00, value]
    lo_bounds = [5.29185155e+02, -4.02435791e+00, -1.43578351e-01]
    hi_bounds = [1.43406714e+03, -2.76718364e+00,  2.87000918e-01]

    expected_texs = [
        '$[529.2, 1434.1]$',
        '$-3.26^{+0.49}_{-0.77}$',
        '$[-0.14, 0.29]$',
    ]
    tex_vals = mu.tex_parameters(values, lo_bounds, hi_bounds)
    for tex, expected in zip(tex_vals, expected_texs):
        assert tex == expected

