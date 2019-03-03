import sys
import os
import random

import numpy as np
import numpy.testing as nt

ROOT = os.path.realpath(os.path.dirname(__file__) + '/..') + '/'
sys.path.append(ROOT)
import MCcubed as mc3
import MCcubed.utils as mu

os.chdir(ROOT + 'tests')


def test_parray_none():
    assert mu.parray('None') is None


def test_parray_empty():
    nt.assert_equal(mu.parray(''), np.array([]))


def test_parray_numbers():
    nt.assert_equal(mu.parray('1 2 3'), np.array([1.0, 2.0, 3.0], np.double))


def test_parray_strings():
    assert mu.parray("a b\nc") == ["a", "b", "c"]


def test_saveascii():
    a = np.arange(4) * np.pi
    b = np.arange(4)
    c = np.logspace(0, 12, 4)
    data = [a,b,c]
    asciifile = "saved_ascii.txt"
    mu.saveascii(data, asciifile)

    with open(asciifile, "r") as f:
        output = f.read()
    assert output == ('        0         0         1\n'
                      '3.1415927         1     10000\n'
                      '6.2831853         2     1e+08\n'
                      ' 9.424778         3     1e+12\n')

def test_loadascii():
    # loadascii() is equivalent to np.loadtxt(file, unpack=True)
    asciifile = "saved_ascii.txt"
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
    nt.assert_equal(data, mu.loadascii(asciifile))


def test_loadsavebin():
    # This could be replaced with pickle files
    binfile = "saved_bin.npz"
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
    nt.assert_equal(outdata[0], np.arange(4))
    assert outdata[1] == 'one'
    nt.assert_equal(outdata[2], np.ones((2,2)))
    assert outdata[3] == True
    assert outdata[4] == [42]
    assert outdata[5] == (42,42)


def test_isfile():
    pass


def test_credregion():
    np.random.seed(2)
    posterior = np.random.normal(0, 1.0, 100000)
    pdf, xpdf, HPDmin = mu.credregion(posterior)
    nt.assert_approx_equal(np.amin(xpdf[pdf>HPDmin]), -1.0, significant=3)
    nt.assert_approx_equal(np.amax(xpdf[pdf>HPDmin]),  1.0, significant=3)


def test_parnames():
    nt.assert_equal(mu.default_parnames(3),
                    np.array(['Param 1', 'Param 2', 'Param 3']))#, dtype='<U7'))
