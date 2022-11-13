#!/bin/bash

# multibuild tooling says "Python should be on the PATH"
python3 -V
ln -s /usr/local/bin/python3 /usr/local/bin/python
export PATH=$PATH:/usr/local/bin
python -V

#brew update
#brew outdated cmake || brew upgrade cmake
git clone --recursive https://github.com/MacPython/terryfy
git clone https://github.com/matthew-brett/multibuild
source multibuild/osx_utils.sh
get_macpython_environment $VERSION venv
pip install wheel
