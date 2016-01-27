from numpy import get_include
import os, re, sys
from distutils.core import setup, Extension

sys.path.append("./../")
import VERSION as ver

srcdir = 'src/'      # C-code source folder
incdir = 'include/'  # Include filder with header files

files = os.listdir(srcdir)
# This will filter the results for just the c files:
files = list(filter(lambda x:     re.search('.+[.]c$',     x), files))
files = list(filter(lambda x: not re.search('[.#].+[.]c$', x), files))

inc = [get_include(), incdir]
eca = []  # ['-fopenmp']
ela = []  # ['-lgomp']

extensions = []
for i in range(len(files)):
  e = Extension(files[i].rstrip(".c"),
                sources=["{:s}{:s}".format(srcdir,files[i])],
                include_dirs=inc,
                extra_compile_args=eca,
                extra_link_args=ela)
  extensions.append(e)


setup(name         = "MC3",
      version      = "{:d}.{:d}.{:d}".format(ver.MC3_VER, ver.MC3_MIN,
                                             ver.MC3_REV),
      author       = "Patricio Cubillos",
      author_email = "patricio.cubillos@oeaw.ac.at",
      url          = "https://github.com/pcubillos/MCcubed",
      description  = "Multi-core Markov-chain Monte Carlo",
      ext_modules  = extensions)
