# Copyright (c) 2015-2021 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

import os
import re
import sys
import setuptools
from setuptools import setup, Extension

from numpy import get_include

sys.path.append(os.path.join(os.path.dirname(__file__), 'mc3'))
from VERSION import __version__


srcdir = 'src_c/'          # C-code source folder
incdir = 'src_c/include/'  # Include folder with header files

cfiles = os.listdir(srcdir)
cfiles = list(filter(lambda x: re.search('.+[.]c$', x), cfiles))
cfiles = list(filter(lambda x: not re.search('[.#].+[.]c$', x), cfiles))

inc = [get_include(), incdir]
eca = ['-ffast-math']
ela = []

extensions = [
    Extension(
        'mc3.lib.' + cfile.rstrip('.c'),
        sources=[f'{srcdir}{cfile}'],
        include_dirs=inc,
        extra_compile_args=eca,
        extra_link_args=ela)
    for cfile in cfiles
    ]


with open('README.md', 'r') as f:
    readme = f.read()

setup(
    name = 'mc3',
    version = __version__,
    author = 'Patricio Cubillos',
    author_email = 'patricio.cubillos@oeaw.ac.at',
    url = 'https://github.com/pcubillos/mc3',
    packages = setuptools.find_packages(),
    install_requires = [
        'numpy>=1.15.0',
        'scipy>=0.17.1',
        'matplotlib>=2.0',
    ],
    tests_require = [
        'pytest>=3.9',
        'dynesty>=0.9.5',
    ],
    include_package_data=True,
    license = 'MIT',
    description = 'Multi-core Markov-chain Monte Carlo package.',
    long_description=readme,
    long_description_content_type='text/markdown',
    include_dirs = inc,
    entry_points={'console_scripts': ['mc3 = mc3.__main__:main']},
    ext_modules = extensions,
    )
