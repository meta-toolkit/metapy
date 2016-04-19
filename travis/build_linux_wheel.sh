#!/bin/bash
# based heavily on https://github.com/matthew-brett/manylinux-builds
set -eo pipefail

# UNICODE_WIDTH selects "32"=wide (UCS4) or "16"=narrow (UTF-16) builds
UNICODE_WIDTH="${UNICODE_WIDTH:-32}"

# Install cmake
wget --no-check-certificate http://www.cmake.org/files/v3.2/cmake-3.2.0-Linux-x86_64.sh
sh cmake-3.2.0-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir

# Install zlib
yum install -y zlib-devel

# taken from https://github.com/matthew-brett/manylinux-builds/blob/master/common_vars.sh
function lex_ver {
    # Echoes dot-separated version string padded with zeros
    # Thus:
    # 3.2.1 -> 003002001
    # 3     -> 003000000
    echo $1 | awk -F "." '{printf "%03d%03d%03d", $1, $2, $3}'
}

# taken from https://github.com/matthew-brett/manylinux-builds/blob/master/common_vars.sh
function strip_dots {
    # Strip "." characters from string
    echo $1 | sed "s/\.//g"
}

# taken from https://github.com/matthew-brett/manylinux-builds/blob/master/common_vars.sh
function cpython_path {
    # Return path to cpython given
    # * version (of form "2.7")
    # * u_width ("16" or "32" default "32")
    #
    # For back-compatibility "u" as u_width also means "32"
    local py_ver="${1:-2.7}"
    local u_width="${2:-${UNICODE_WIDTH}}"
    local u_suff=u
    # Back-compatibility
    if [ "$u_width" == "u" ]; then u_width=32; fi
    # For Python >= 3.3, "u" suffix not meaningful
    if [ $(lex_ver $py_ver) -ge $(lex_ver 3.3) ] ||
        [ "$u_width" == "16" ]; then
        u_suff=""
    elif [ "$u_width" != "32" ]; then
        echo "Incorrect u_width value $u_width"
        # exit 1
    fi
    local no_dots=$(strip_dots $py_ver)
    echo "/opt/python/cp${no_dots}-cp${no_dots}m${u_suff}"
}

# taken from https://github.com/matthew-brett/manylinux-builds/blob/master/common_vars.sh
function repair_wheelhouse {
    local in_dir=$1
    local out_dir=$2
    for whl in $in_dir/*.whl; do
        if [[ $whl == *none-any.whl ]]; then
            cp $whl $out_dir
        else
            auditwheel repair $whl -w $out_dir/
        fi
    done
    chmod -R a+rwX $out_dir
}

# adapted from https://github.com/matthew-brett/manylinux-builds/blob/master/build_sklearns.sh
PIP="$(cpython_path $PYTHON_VERSION)/bin/pip"
pushd /metapy
$PIP wheel -w unfixed_wheels --verbose ./
ls unfixed_wheels/*.whl
repair_wheelhouse unfixed_wheels dist
ls dist/*.whl
$PIP install dist/*.whl
popd
