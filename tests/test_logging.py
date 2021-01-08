# Copyright (c) 2015-2021 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

import pytest
import mc3.utils as mu


log_content = ["Debugging",
"Hello, this is log",
"Headline",
"""
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  Warning:
    Warning!
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

""",
"""::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  Error in module: 'test_logging.py', function: 'test_log_error', line: 52
    Error
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
"""]


@pytest.mark.parametrize('verb', [3, 2, 1, 0, -1])
def test_log(tmp_path, verb):
    log_file = str(tmp_path / 'test.log')

    log = mu.Log(log_file, verb=verb)
    log.debug('Debugging')
    log.msg('Hello, this is log')
    log.head('Headline')
    log.warning('Warning!')
    log.close()

    with open(log_file, 'r') as f:
        content = f.read()
    assert content == '\n'.join(log_content[3-verb:4])


def test_log_error(tmp_path):
    verb = 2
    log_file = str(tmp_path / 'test.log')

    log = mu.Log(log_file, verb=verb)
    log.debug('Debugging')
    log.msg('Hello, this is log')
    log.head('Headline')
    log.warning('Warning!')

    with pytest.raises(SystemExit):
        log.error('Error')  # Line number in log_content must match this one

    with open(log_file, 'r') as f:
        content = f.read()
    assert content == '\n'.join(log_content[3-verb:])


def test_context_manager(capsys):
    msg = 'Hello, this is log'
    with mu.Log() as log:
        log.msg(msg)
    captured = capsys.readouterr()
    assert captured.out == msg + '\n'


def test_tracklev(tmp_path):
    verb = 2
    log_file = str(tmp_path / 'test.log')

    log = mu.Log(log_file, verb=verb)
    with pytest.raises(SystemExit):
        log.error('Error', tracklev=1)

    with open(log_file, 'r') as f:
        content = f.read()
    assert "Error in module: '__init__.py', function: 'main', line:" in content

