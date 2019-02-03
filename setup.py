import os
import sys
import re
import setuptools
from numpy import get_include
from setuptools import setup, Extension

topdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(topdir + "/MCcubed")
import VERSION as ver


version = "{:d}.{:d}.{:d}".format(ver.MC3_VER, ver.MC3_MIN, ver.MC3_REV)

srcdir = topdir + '/src_c/'          # C-code source folder
incdir = topdir + '/src_c/include/'  # Include filder with header files

files = os.listdir(srcdir)
# This will filter the results for just the c files:
files = list(filter(lambda x:     re.search('.+[.]c$',     x), files))
files = list(filter(lambda x: not re.search('[.#].+[.]c$', x), files))

inc = [get_include(), incdir]
eca = []
ela = []

extensions = []
for i in range(len(files)):
    e = Extension("MCcubed.lib."+files[i].rstrip(".c"),
                  sources=["{:s}{:s}".format(srcdir, files[i])],
                  include_dirs=inc,
                  extra_compile_args=eca,
                  extra_link_args=ela)
    extensions.append(e)


setup(name         = "MCcubed",
      version      = version,
      author       = "Patricio Cubillos",
      author_email = "patricio.cubillos@oeaw.ac.at",
      url          = "https://github.com/pcubillos/MCcubed",
      packages     = setuptools.find_packages(),
      install_requires = ['numpy>=1.13.3',
                          'scipy>=0.17.1',
                          'matplotlib>=2.2.3',],
      license      = "MIT",
      description  = "Multi-core Markov-chain Monte Carlo package.",
      include_dirs = inc,
      #entry_points={"console_scripts": ['foo = MCcubed.mccubed:main']},
      ext_modules  = extensions)
