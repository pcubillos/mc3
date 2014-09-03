from numpy import get_include
import os, re, sys
from distutils.core import setup, Extension

srcdir = 'src/'  # C-code source folder
libdir = 'lib/'  # Where the shared objects are put

files = os.listdir(srcdir)
# This will filter the results for just the c files:
files = filter(lambda x:     re.search('.+[.]c$',     x), files)
files = filter(lambda x: not re.search('[.#].+[.]c$', x), files)

ext_mod = []
inc = [get_include()]

for i in range(len(files)):
  exec("mod{:d} = Extension('{:s}', sources=['{:s}{:s}'], include_dirs=inc, "
       "extra_compile_args=['-fopenmp'], extra_link_args=['-lgomp'])".format(
       i, files[i].rstrip('.c'), srcdir, files[i]))

  exec('ext_mod.append(mod{:d})'.format(i))

setup(name=libdir, version='1.0', description='c extension functions',
      ext_modules = ext_mod)
