#!/bin/bash

# Apparently, Apple moved the path to these in Big Sur, confusing built-in configuration
export CLANG_SYSROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
export LIBRARY_PATH="${CLANG_SYSROOT}/usr/lib"
export LINK_FLAGS="-isysroot ${CLANG_SYSROOT}"
export CFLAGS="-isysroot ${CLANG_SYSROOT}"
export CXXFLAGS="-isysroot ${CLANG_SYSROOT}"
export CPPFLAGS="-isysroot ${CLANG_SYSROOT}"

pip wheel -w dist --verbose ./
ls dist/*.whl
pip install dist/*.whl
