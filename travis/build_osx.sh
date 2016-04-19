#!/bin/bash
pip wheel -w dist --verbose ./
ls dist/*.whl
pip install dist/*.whl
