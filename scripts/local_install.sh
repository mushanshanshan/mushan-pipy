#!/usr/bin/env bash

pip uninstall mushan -y

python setup.py clean
python setup.py install
