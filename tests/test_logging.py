import pytest

import MCcubed.utils as mu


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
  Error in module: 'test_logging.py', function: 'test_log_error', line: 50
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
        log.error('Error')

    with open(log_file, 'r') as f:
        content = f.read()
    assert content == '\n'.join(log_content[3-verb:])


def test_tracklev(tmp_path):
    verb = 2
    log_file = str(tmp_path / 'test.log')

    log = mu.Log(log_file, verb=verb)
    with pytest.raises(SystemExit):
        log.error('Error', tracklev=1)

    with open(log_file, 'r') as f:
        content = f.read()
    assert "Error in module: '__init__.py', function: 'main', line:" in content
