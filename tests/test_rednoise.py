# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

import pytest
import numpy as np
import MCcubed.rednoise as rednoise


np.random.seed(12)
N = 1000
white = np.random.normal(0, 5, N)
red = np.sin(np.arange(N)/(0.1*N))*np.random.normal(1.0, 1.0, N)
data = white + red


expected_red_rms = np.array(
    [5.20512494, 2.36785563, 1.72466452, 1.49355819, 1.52934937,
     1.35774105, 1.11881588, 1.13753563, 1.16566184, 1.03510878,
     1.11692786, 0.95551055, 1.04041202, 0.86876758, 0.93962365,
     0.95093077, 0.86283389, 0.89332354, 0.95500342, 0.82927083])

expected_red_rmslo = np.array(
    [0.11639013, 0.12995296, 0.1285489 , 0.13412548, 0.15774034,
     0.15574358, 0.13041103, 0.14351302, 0.1550736 , 0.14721337,
     0.16700106, 0.15015152, 0.1685249 , 0.14533717, 0.1627079 ,
     0.16987469, 0.1604309 , 0.17348578, 0.19451647, 0.17348533])

expected_red_rmshi = np.array(
    [0.11639013, 0.12995296, 0.1285489 , 0.13412548, 0.15774034,
     0.15574358, 0.1611256 , 0.18169027, 0.20020244, 0.19264249,
     0.22147211, 0.20384028, 0.23076986, 0.2007309 , 0.22759927,
     0.24306181, 0.23335404, 0.25645724, 0.29446565, 0.26262799])

expected_red_stderr = np.array(
    [5.20664133, 2.13096763, 1.57786671, 1.31163   , 1.14789132,
     1.03429558, 0.94962841, 0.8838618 , 0.83021424, 0.78624182,
     0.74867937, 0.71682123, 0.68816067, 0.66305576, 0.64091963,
     0.62131904, 0.60393775, 0.58855564, 0.57504053, 0.55986528])

expected_binsz = np.array(
    [ 1.,  6., 11., 16., 21., 26., 31., 36., 41., 46., 51., 56., 61.,
     66., 71., 76., 81., 86., 91., 96.])

expected_white_rms = np.array(
    [5.13108371, 2.24264189, 1.54890969, 1.32144868, 1.3520051 ,
     1.16925098, 0.88639028, 0.91812782, 0.93234654, 0.8127796 ,
     0.86662417, 0.7447655 , 0.81963664, 0.68330918, 0.65699017,
     0.73730708, 0.62304519, 0.65482596, 0.7385728 , 0.60835201])

expected_white_rmslo = np.array(
    [0.11473452, 0.12308096, 0.11544891, 0.11866959, 0.13944868,
     0.13412229, 0.10331912, 0.11583223, 0.12403454, 0.11559367,
     0.1295761 , 0.11703448, 0.13276393, 0.11431161, 0.11376628,
     0.13171286, 0.11584582, 0.12716893, 0.15043357, 0.12726862])

expected_white_rmshi = np.array(
    [0.11473452, 0.12308096, 0.11544891, 0.11866959, 0.13944868,
     0.13412229, 0.12765297, 0.14664586, 0.16013053, 0.15126515,
     0.17184018, 0.15888177, 0.18180051, 0.15788028, 0.15913869,
     0.18845872, 0.16850302, 0.18798885, 0.22773145, 0.19266356])

expected_white_stderr = np.array(
    [5.13332205, 2.1009596 , 1.55564739, 1.29315979, 1.13172685,
     1.01973075, 0.93625586, 0.87141536, 0.81852327, 0.77517006,
     0.73813656, 0.70672705, 0.67847008, 0.65371869, 0.63189428,
     0.6125697 , 0.59543317, 0.58026767, 0.56694288, 0.55198132])


def test_binrms_values_red():
    rms, rmslo, rmshi, stderr, binsz = rednoise.binrms(data, len(data)/10, 5)
    np.testing.assert_almost_equal(rms,    expected_red_rms)
    np.testing.assert_almost_equal(rmslo,  expected_red_rmslo)
    np.testing.assert_almost_equal(rmshi,  expected_red_rmshi)
    np.testing.assert_almost_equal(stderr, expected_red_stderr)
    np.testing.assert_almost_equal(binsz,  expected_binsz)


def test_binrms_values_white():
    rms, rmslo, rmshi, stderr, binsz = rednoise.binrms(white, len(data)/10, 5)
    np.testing.assert_almost_equal(rms,    expected_white_rms)
    np.testing.assert_almost_equal(rmslo,  expected_white_rmslo)
    np.testing.assert_almost_equal(rmshi,  expected_white_rmshi)
    np.testing.assert_almost_equal(stderr, expected_white_stderr)
    np.testing.assert_almost_equal(binsz,  expected_binsz)


def test_binrms_defaults():
    rms, rmslo, rmshi, stderr, binsz = rednoise.binrms(data)
    assert len(rms)    == 500
    assert len(rmslo)  == 500
    assert len(rmshi)  == 500
    assert len(stderr) == 500
    assert len(binsz)  == 500


@pytest.mark.parametrize('maxbins', [200, 200.0])
def test_binrms_maxbins(maxbins):
    rms, rmslo, rmshi, stderr, binsz = rednoise.binrms(data, maxbins)
    assert True


@pytest.mark.parametrize('binstep', [1, 1.0, 2, 2.0])
def test_binrms_binstep(binstep):
    maxbins = len(data) // 2
    rms, rmslo, rmshi, stderr, binsz = rednoise.binrms(data, maxbins, binstep)
    assert len(rms) == len(data) // binstep // 2


@pytest.mark.parametrize('dtype', [tuple, list, np.array])
def test_binrms_data_type(dtype):
    rms, rmslo, rmshi, stderr, binsz = rednoise.binrms(dtype(data))
    assert True
