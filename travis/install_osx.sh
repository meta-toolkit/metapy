#!/bin/bash
brew update
brew outdated cmake || brew upgrade cmake
git clone --recursive https://github.com/MacPython/terryfy
source terryfy/travis_tools.sh
get_python_environment macpython $VERSION venv
pip install wheel
