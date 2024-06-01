# Copyright (c) 2015-2023 Patricio Cubillos and contributors.
# mc3 is open-source software under the MIT license (see LICENSE).

import os
import re
from setuptools import setup, Extension

from numpy import get_include


# C-code source and include folders
srcdir = 'src_c/'
incdir = 'src_c/include/'

cfiles = os.listdir(srcdir)
cfiles = list(filter(lambda x: re.search('.+[.]c$', x), cfiles))
cfiles = list(filter(lambda x: not re.search('[.#].+[.]c$', x), cfiles))

inc = [get_include(), incdir]
eca = ['-lm', '-O3', '-ffast-math']
ela = ['-lm']

extensions = [
    Extension(
        'mc3.lib.' + cfile.rstrip('.c'),
        sources=[f'{srcdir}{cfile}'],
        include_dirs=inc,
        extra_compile_args=eca,
        extra_link_args=ela,
    )
    for cfile in cfiles
]


setup(
    ext_modules = extensions,
    include_dirs = inc,
)
