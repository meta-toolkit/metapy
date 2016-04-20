#!/bin/bash
sudo docker run --rm \
    -e PYTHON_VERSION=$VERSION \
    -e UNICODE_WIDTH=$UNICODE_WIDTH \
    -v `pwd`:/metapy \
    quay.io/pypa/manylinux1_x86_64 /metapy/travis/build_linux_wheel.sh
