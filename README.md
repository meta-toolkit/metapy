# metapy: (experimental) Python bindings for [MeTA][meta]

[![Build Status](https://travis-ci.org/meta-toolkit/metapy.svg?branch=master)](https://travis-ci.org/meta-toolkit/metapy)

[![Windows Build Status](https://ci.appveyor.com//api/projects/status/github/meta-toolkit/metapy?svg=true&branch=master)](https://ci.appveyor.com/project/skystrife/metapy)

This project provides Python (2.7 and 3.x are supported) bindings for the
MeTA toolkit. They are still very much under construction, but the goal is
to make it seamless to use MeTA's components within any Python application
(e.g., a Django or Flask web app).

This project is made possible by the excellent [pybind11][pybind11]
library.

## Getting Started (the easy way)

```bash
# Ensure your pip is up to date
pip install --upgrade pip

# install metapy!
pip install metapy
```

This should work on Linux, OS X, and Windows with pretty much any recent
Python version >= 2.7. On Linux, make sure to update your `pip` to version
8.1 so you can install from a binary package---this will save you a lot of
time.

## Getting Started (the hard way)

You will, of course, need Python installed. You will also need its headers
to be installed as well, so look for a `python-dev` or similar package for
your system. Beyond that, you'll of course need to satisfy the requirements
for [building MeTA itself][build-guide].

This repository should have everything you need to get started. You should
ensure that you've fetched all of the submodules first, though:

```bash
git submodule update --init --recursive
```

Once that's done, you should be able to build the library like so:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

You can force building against a specific version of Python if you happen
to have multiple versions installed by specifying
`-DMETAPY_PYTHON_VERSION=x.y` when invoking `cmake`.

The module should be written to `metapy.so` in the build directory.

[meta]: https://meta-toolkit.org
[pybind11]: https://github.com/pybind/pybind11
[build-guide]: https://meta-toolkit.org/setup-guide.html
