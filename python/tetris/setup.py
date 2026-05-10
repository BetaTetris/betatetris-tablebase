#!/usr/bin/env python3

import tempfile
from pathlib import Path
from setuptools import Extension, setup
from setuptools.errors import CompileError, LinkError
from setuptools.command.build_ext import build_ext

import numpy

sources = ['tetris.cpp', 'state.cpp', 'board.cpp', 'supervised_reader.cpp', 'module.cpp',
           '../../src/tetris.cpp', '../../src/frame_sequence.cpp']
NAME = 'tetris'

class build_ext_ex(build_ext):
    extra_compile_args = {
        'tetris': {
            'unix': ['-std=c++20', '-DLINE_CAP=430', '-mbmi2'],
            #'unix': ['-std=c++20', '-DLINE_CAP=290', '-mbmi2', '-DNO_2KS', '-DTETRIS_ONLY'],
            #'unix': ['-std=c++20', '-mbmi2', '-DNO_ROTATION'],
            #'unix': ['-std=c++20', '-DLINE_CAP=430', '-mbmi2', '-fsanitize=address', '-fsanitize=undefined', '-O1'],
            'msvc': ['/std:c++20', '/DLINE_CAP=430', '/DADJ_DELAY=18', '/DTAP_SPEED=Tap30Hz'],
        }
    }

    def _zstd_found(self) -> bool:
        code = r"""
        int ZSTD_versionNumber(void);
        int main(void) {
            return ZSTD_versionNumber() == 0;
        }
        """
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src = td_path / "check_zstd.c"
            src.write_text(code, encoding="utf-8")
            try:
                objects = self.compiler.compile([str(src)], output_dir=str(td_path))
                self.compiler.link_executable(
                    objects,
                    str(td_path / "check_zstd"),
                    libraries=["zstd"],
                )
            except (CompileError, LinkError, OSError):
                return False
        return True

    def build_extension(self, ext):
        zstd_found = self._zstd_found()
        if zstd_found:
            ext.libraries = list(ext.libraries or []) + ["zstd"]
        else:
            print("zstd not found; you won't be able to run training with supervised data")

        extra_args = self.extra_compile_args.get(ext.name)
        if extra_args is not None:
            ctype = self.compiler.compiler_type
            ext.extra_compile_args = extra_args.get(ctype, [])
            ext.extra_link_args = extra_args.get(ctype, [])

        build_ext.build_extension(self, ext)

module = Extension(
    NAME,
    sources=sources,
    libraries=[],
    include_dirs=[numpy.get_include()],
)
setup(name=NAME, ext_modules=[module], cmdclass={'build_ext': build_ext_ex})
