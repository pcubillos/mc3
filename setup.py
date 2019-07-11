# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

import os
import re
import setuptools
from setuptools import setup, Extension

from numpy import get_include

import sys
sys.path.append('MCcubed')
import VERSION as v
version = "{:d}.{:d}.{:d}".format(v.MC3_VER, v.MC3_MIN, v.MC3_REV)

srcdir = 'src_c/'          # C-code source folder
incdir = 'src_c/include/'  # Include folder with header files

cfiles = os.listdir(srcdir)
cfiles = list(filter(lambda x:     re.search('.+[.]c$',     x), cfiles))
cfiles = list(filter(lambda x: not re.search('[.#].+[.]c$', x), cfiles))

inc = [get_include(), incdir]
eca = ['-ffast-math']
ela = []

extensions = []
for cfile in cfiles:
    e = Extension("MCcubed.lib." + cfile.rstrip(".c"),
        sources=["{:s}{:s}".format(srcdir, cfile)],
        include_dirs=inc,
        extra_compile_args=eca,
        extra_link_args=ela)
    extensions.append(e)

with open('README.md', 'r') as f:
    readme = f.read()

setup(name         = "mc3",
      version      = version,
      author       = "Patricio Cubillos",
      author_email = "patricio.cubillos@oeaw.ac.at",
      url          = "https://github.com/pcubillos/MCcubed",
      packages     = setuptools.find_packages(),
      install_requires = ['numpy>=1.13.3',
                          'scipy>=0.17.1',
                          'matplotlib>=2.2.3',],
      tests_require = ['pytest>=3.9',],
      include_package_data=True,
      license      = "MIT",
      description  = "Multi-core Markov-chain Monte Carlo package.",
      long_description=readme,
      long_description_content_type="text/markdown",
      include_dirs = inc,
      entry_points={"console_scripts": ['mc3 = MCcubed.__main__:main']},
      ext_modules  = extensions)
