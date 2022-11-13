#!/bin/bash

# Apparently, Apple moved the path to these in Big Sur, confusing built-in configuration
export CLANG_SYSROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
export LIBRARY_PATH="${CLANG_SYSROOT}/usr/lib"
export LINK_FLAGS="-isysroot ${CLANG_SYSROOT}"
export CFLAGS="-isysroot ${CLANG_SYSROOT}"
export CXXFLAGS="-isysroot ${CLANG_SYSROOT}"
export CPPFLAGS="-isysroot ${CLANG_SYSROOT}"

# We want to set the minimum target to macOS 11 (~= 10.16, Big Sur)
# to account for very old Pythons that don't know about anything newer.
# For example, the default Python 3.7 that this tooling installs is Python 3.7.9
# and that doesn't understand anything after 10.X. That sets the minimum
# floor for Python version to 3.7 and macOS verion to 11, but there is good
# coverage in the original project for anything older: we'll just concentrate on newer.
export MACOSX_DEPLOYMENT_TARGET=10.16

pip wheel -w dist --verbose ./
ls dist/*.whl
pip install dist/*.whl
