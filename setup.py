#!/usr/bin/env python
from __future__ import print_function

import contextlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile

from setuptools import setup, Extension
from setuptools.command import build_ext, bdist_egg, develop
from distutils.spawn import find_executable
from distutils import log, sysconfig
from distutils.command import build

VERSION = "0.0.1.dev11"

# See:
# http://stackoverflow.com/questions/3223604/how-to-create-a-temporary-directory-and-get-the-path-file-name-in-python
@contextlib.contextmanager
def cd(newdir, cleanup=lambda: True):
    previdr = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(previdr)
        cleanup()

@contextlib.contextmanager
def tempdir():
    dirpath = tempfile.mkdtemp()
    def cleanup():
        shutil.rmtree(dirpath)
    with cd(dirpath, cleanup):
        yield dirpath

# Based on https://github.com/symengine/symengine.py/blob/master/setup.py
class CMakeBuildExt(build.build):
    user_options = build.build.user_options + \
            [('icu-root=', None, "Path to ICU root"),
             ('generator=', None, "CMake build generator")]

    def initialize_options(self):
        build.build.initialize_options(self)
        self.icu_root = None
        self.generator = 'MSYS Makefiles' if platform.system() == 'Windows' else None

    def cmake_build(self):
        src_dir = os.path.dirname(os.path.realpath(__file__))

        cmake_exe = find_executable("cmake")
        if not cmake_exe:
            raise EnvironmentError("Could not find cmake executable")

        py_version = "{}.{}".format(sys.version_info[0], sys.version_info[1])
        cmake_cmd = [cmake_exe, src_dir, "-DCMAKE_BUILD_TYPE=Release"]

        if platform.system() == 'Windows':
            cmake_cmd.append("-DMETAPY_PYTHON_VERSION={}".format(py_version))
        else:
            cmake_cmd.append("-DPYTHON_INCLUDE_DIRS={}".format(sysconfig.get_python_inc()))

        if self.icu_root:
            cmake_cmd.extend(["-DICU_ROOT={}".format(self.icu_root)])

        if self.generator:
            cmake_cmd.extend(["-G{}".format(self.generator)])

        with tempdir() as dirpath:
            print("Build directory: {}".format(os.getcwd()))
            if subprocess.call(cmake_cmd) != 0:
                raise EnvironmentError("CMake invocation failed")

            if subprocess.call([cmake_exe, "--build", "."]) != 0:
                raise EnvironmentError("CMake build failed")

            if subprocess.call([cmake_exe, "--build", ".", "--target",
                "install"]) != 0:
                raise EnvironmentError("CMake install failed")

        # Make dummy __init__.py
        initpy = os.path.join(src_dir, "dist", "metapy", "__init__.py")

        with open(initpy, "w") as f:
            f.write("from .metapy import *\n")
            f.write('__version__ = "{}"\n'.format(VERSION))

        # Copy over extra DLLs on Windows
        if platform.system() == 'Windows':
            dlls = ['libwinpthread-1.dll', 'libgcc_s_seh-1.dll', 'libstdc++-6.dll', 'zlib1.dll']
            for dll in dlls:
                shutil.copyfile(os.path.join("c:", os.sep, "msys64", "mingw64", "bin", dll),
                                os.path.join(src_dir, "dist", "metapy", dll))

    def run(self):
        self.cmake_build()
        return build.build.run(self)

class DummyBuildExt(build_ext.build_ext):
    def __init__(self, *args, **kwargs):
        build_ext.build_ext.__init__(self, *args, **kwargs)

    def run(self):
        # do nothing; cmake already built the extension
        pass

class DummyBDistEgg(bdist_egg.bdist_egg):
    def __init__(self, *args, **kwargs):
        bdist_egg.bdist_egg.__init__(self, *args, **kwargs)

    def run(self):
        self.run_command("build")
        return bdist_egg.bdist_egg.run(self)

class DummyDevelop(develop.develop):
    def __init__(self, *args, **kwargs):
        develop.develop.__init__(self, *args, **kwargs)

    def run(self):
        self.run_command("build")
        return develop.develop.run(self)

def clean_dist():
    src_dir = os.path.dirname(os.path.realpath(__file__))

    dist_dir = os.path.join(src_dir, "dist", "metapy")
    if os.path.exists(dist_dir):
        log.info("Deleting distribution directory {}".format(dist_dir))
        shutil.rmtree(dist_dir)

    os.makedirs(dist_dir)

clean_dist()

setup(name = 'metapy',
      version = VERSION,
      description = 'Python bindings for MeTA',
      author = 'Chase Geigle',
      author_email = 'geigle1@illinois.edu',
      url = 'https://github.com/meta-toolkit/metapy',
      license = 'MIT',
      packages = ['metapy'],
      package_dir = { '': 'dist' },
      include_package_data = True,
      cmdclass = {
          'build': CMakeBuildExt,
          'build_ext': DummyBuildExt,
          'bdist_egg': DummyBDistEgg,
          'develop':   DummyDevelop
      },
      zip_safe = False,
      ext_modules = [Extension('metapy', [])],
      ext_package='metapy',
      classifiers = [
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'License :: OSI Approved :: University of Illinois/NCSA Open Source License',
          'Operating System :: POSIX :: Linux',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Text Processing',
          'Topic :: Text Processing :: Filters',
          'Topic :: Text Processing :: General',
          'Topic :: Text Processing :: Indexing',
          'Topic :: Text Processing :: Linguistic',
      ],
      keywords = [
          'NLP', 'natural language processing',
          'IR', 'information retrieval',
          'CL', 'computational lingusitics',
          'parsing', 'tagging', 'tokenizing', 'syntax', 'lingustics',
          'natural language', 'text mining', 'text analysis'
      ])
