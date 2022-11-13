#!/bin/bash

find . -type d -name "build" | xargs rm -rf
rm -rf venv
rm -rf dist
