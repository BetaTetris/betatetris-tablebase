#!/usr/bin/env python3

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import numpy

sources = ['board.cpp', 'tetris.cpp', 'module.cpp']

class build_ext_ex(build_ext):
    extra_compile_args = {
        'tetris': {
            'unix': ['-std=c++20', '-DLINE_CAP=430', '-DADJ_DELAY=18', '-DTAP_SPEED=Tap30Hz', '-DNO_2KS=1', '-mbmi2'],
            #'unix': ['-std=c++20', '-DLINE_CAP=430', '-DADJ_DELAY=18', '-DTAP_SPEED=Tap30Hz', '-mbmi2', '-O1'],
            #'unix': ['-std=c++20', '-DLINE_CAP=430', '-DADJ_DELAY=18', '-DTAP_SPEED=Tap30Hz', '-mbmi2', '-fsanitize=address', '-fsanitize=undefined', '-O1'],
            'msvc': ['/std:c++20', '/DLINE_CAP=430', '/DADJ_DELAY=18', '/DTAP_SPEED=Tap30Hz', '/NO_2KS=1'],
        }
    }

    def build_extension(self, ext):
        extra_args = self.extra_compile_args.get(ext.name)
        if extra_args is not None:
            ctype = self.compiler.compiler_type
            ext.extra_compile_args = extra_args.get(ctype, [])
            ext.extra_link_args = extra_args.get(ctype, [])

        build_ext.build_extension(self, ext)

name = 'tetris'
module = Extension(
    name,
    sources=sources,
    include_dirs=[numpy.get_include()],
)
setup(name=name, ext_modules=[module], cmdclass={'build_ext': build_ext_ex})
